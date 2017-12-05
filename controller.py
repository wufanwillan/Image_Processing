import curses
import config
import rospy
import cv_bridge
import end2end
from sets import Set
from policy import Policy
from sensor_msgs.msg import Image
from std_msgs.msg import String
from geometry_msgs.msg import Twist
from threading import Thread
from collections import deque

cvbridge = cv_bridge.CvBridge()
class TextWindow(object):
    _screen = None
    _window = None
    _num_lines = None

    def __init__(self, stdscr, lines=16):
        self._screen = stdscr
        self._screen.nodelay(False)
        self._num_lines = lines
        curses.curs_set(0)

    def read_key(self):
        key = self._screen.getkey()
        return key

    def clear(self):
        self._screen.clear()

    def write_line(self, lineno, message):
        if lineno < 0 or lineno > self._num_lines:
            raise ValueError, 'lineno out of bounds'
        height, width = self._screen.getmaxyx()
        y = (height / self._num_lines) * lineno
        x = 10
        for text in message.split('\n'):
            text = text.ljust(width)
            self._screen.addstr(y, x, text)
            y += 1

    def refresh(self):
        self._screen.refresh()

    def beep(self):
        curses.flash()

class Controller(object):
    def __init__(self, win):
        self.win = win
        self.rate = rospy.Rate(config.RATE)
        self.pub = rospy.Publisher(config.PUBLISH_TOPIC[0], config.PUBLISH_TOPIC[1], queue_size = config.PUBLISH_TOPIC[2])
        rospy.Subscriber(config.SUBSCRIBE_TOPIC[0], config.SUBSCRIBE_TOPIC[1], self.cb_image, queue_size=config.SUBSCRIBE_TOPIC[2], buff_size=2**24)
        rospy.Subscriber('/intention', String, self.cb_intention)

        self.key = None
        self.control_switches={}
        self.current_control = None
        self.exiting = False

        self.log = ''
        self.latency=0
        self.training=False
        self.twist=Twist()
        self.image_msg = None
        self.image_msgs = deque()
        # default intention
        self.intention = config.FORWARD
        print 'controller initialized'

    def cb_intention(self, msg):
        self.intention = msg.data

    def cb_image(self, msg):
        self.image_msg = msg
        # maintain deque of k frames
        self.image_msgs.append(msg)
        if len(self.image_msgs) > config.K_FRAMES:
            self.image_msgs.popleft()

    def get_key_loop(self):
        while not self.exiting:
            self.key = self.win.read_key()

    def register(self, control):
        for k in control.keys:
            self.control_switches[k] = control

    def control_loop(self):
        if self.key == 'q':
            self.exiting = True
            self.publish()
            return
        if self.key == 't':
            self.training = not self.training
            self.key = ''
            return

        if self.key in self.control_switches:
            con = self.control_switches[self.key]
            if con is not self.current_control:
                self.current_control = con

        if self.current_control:
            self.current_control.control_loop(self)
        self.publish()
        return

    def set_twist(self, twist):
        self.twist = twist

    def publish(self):
        win = self.win
        win.clear()
        training_msg = 'training mode' if self.training else 'training disabled'
        win.write_line(1, training_msg)
        win.write_line(2, 'key: %s' % (self.key))
        win.write_line(3, 'intention: %s' % (self.intention))
        win.write_line(4, 'latency, %.4f' % (self.latency))
        control_name = self.current_control.name if self.current_control else 'None'
        win.write_line(5, 'control: %s ~ %s' % (control_name, self.log))
        win.refresh()

        self.pub.publish(self.twist)

    def run(self):
        getkey_thread = Thread(target=self.get_key_loop)
        getkey_thread.start()

        while not self.exiting:
            self.control_loop()
            self.rate.sleep()

        getkey_thread.join()

class AutoControl(object):
     def __init__(self, clf, task, keys='a'):
         self.clf = clf
         self.keys = keys
         self.name = 'Auto Controller %s' % (task)

     def control_loop(self, s):
         msg = s.image_msg
         if msg:
             ts = s.image_msgs[-1].header.stamp
             imgs = [cvbridge.imgmsg_to_cv2(msg, desired_encoding='bgr8') for msg in list(s.image_msgs)]

             twist = self.clf.predict_twist(imgs, s.intention)
             s.set_twist(twist)
             s.latency = (rospy.get_rostime() - ts).to_sec()
             s.log = '\n' + repr(twist)
         else:
             s.log = 'None Image'

class TeleControl(object):
    MOVE_BINDINGS = {
             '8':(2,0),
             'i':(1,0),
             'h':(2,2),
             'j':(1,1),
             ';':(2,-2),
             'l':(1,-1),
             'k':(0,0),
             ',':(-1,0),
             }
    SPEED = 0.5
    TURN = 0.5
    keys = MOVE_BINDINGS.keys()

    def __init__(self):
	# publishing as training data
	self.pub_teleop_vel = rospy.Publisher('train/cmd_vel', Twist, queue_size=1)
	self.pub_image = rospy.Publisher('train/image_raw', Image, queue_size=1)
        self.name = "TeleController"

    def control_loop(self, s):
	k = s.key
	twist = self.make_twist(k)
	s.set_twist(twist)
	s.log = 'teleop got key %s\n' % s.key + repr(twist)

	if s.training:
	    self.publish(s)

    def make_twist(self, k):
	twist = Twist()
        if k in Set(self.keys):
            ds, dt = self.MOVE_BINDINGS[k]
            ds *= self.SPEED
            dt *= self.TURN
            twist.linear.x = ds
            twist.angular.z = dt
	return twist

    def publish(self, s):
        if s.image_msg:
            self.pub_image.publish(s.image_msg)
            self.pub_teleop_vel.publish(s.twist)

def control(stdscr):
    rospy.init_node('controller')
    win = TextWindow(stdscr)
    con = Controller(win)
    con.register(TeleControl())
    clf = Policy(config.TASK)
    con.register(AutoControl(clf, config.TASK, 'a'))

    try:
        con.run()
    finally:
        con.pub.publish(Twist())

if __name__ == '__main__':
    try:
        curses.wrapper(control)
    except rospy.ROSInterruptException:
        pass
