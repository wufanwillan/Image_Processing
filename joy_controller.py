import curses
import config
import rospy
import cv_bridge
import end2end2
from sets import Set
from policy import Policy
from sensor_msgs.msg import Image
from std_msgs.msg import Float32MultiArray, String
from geometry_msgs.msg import Twist
from threading import Thread
from collections import deque
# joy mapping
from sensor_msgs.msg import Joy
from joy_teleop import JOY_MAPPING
from statistics import SmoothStatistics as Stat
import time

import numpy as np
import cv2


# use to measure time
AUTO_START=None
cvbridge = cv_bridge.CvBridge()
class TextWindow(object):
    _screen = None
    _window = None
    _num_lines = None

    def __init__(self, stdscr, lines=20):
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
    tele_twist = Twist()

    def __init__(self, win):
        self.MAX_X = config.MAX_X
        self.MAX_Z = config.MAX_Z
        self.win = win
        self.rate = rospy.Rate(config.RATE)
        self.pub = rospy.Publisher(config.PUBLISH_TOPIC[0], config.PUBLISH_TOPIC[1], queue_size = config.PUBLISH_TOPIC[2])

        self.key = None
        self.control_switches={}
        self.current_control = None
        self.exiting = False

        self.log = ''
        self.latency=0
        self.training=False
        self.reset_stat=False
        self.twist=Twist()
        self.image_msg = None
        self.image_msgs = deque()
        #self.intentions = deque()
        if config.USE_DISCRETE_INTENTION:
            self.intention=config.FORWARD
        else:
            self.intention=[0.0]*config.NUM_INTENTION#config.FORWARD

        rospy.Subscriber(config.SUBSCRIBE_TOPIC[0], config.SUBSCRIBE_TOPIC[1], self.cb_image, queue_size=config.SUBSCRIBE_TOPIC[2], buff_size=2**24)
        if config.USE_DISCRETE_INTENTION:
            #rospy.Subscriber('/train/intention', String, self.cb_intention, queue_size=1)
            rospy.Subscriber('/test_intention', String, self.cb_intention, queue_size=1)
        else:
            rospy.Subscriber('/test_intention', Float32MultiArray, self.cb_intention, queue_size=1)
	rospy.Subscriber("joy", Joy, self.callback)
        print 'controller initialized'

    def cb_intention(self, msg):
        self.intention=msg.data
        '''
        # maintain deque of k frames
        self.intentions.append(msg.data)
        if len(self.intentions) > config.K_FRAMES:
            self.intentions.popleft()
        '''

    def cb_image(self, msg):
        self.image_msg = msg

        # maintain deque of k frames
        self.image_msgs.append(msg)
        if len(self.image_msgs) > config.K_FRAMES:
            self.image_msgs.popleft()

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
        if self.key == 'rst':
            self.reset_stat = not self.reset_stat
            self.key = ''
            return
        # parse intention
        if self.key == 'ii':
            self.intention=config.FORWARD
        if self.key == 'kk':
            self.intention=config.STOP#config.BACKWARD
        if self.key == 'll':
            self.intention=config.LEFT
        if self.key == 'rr':
            self.intention=config.RIGHT

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
        if self.win:
            win = self.win
            win.clear()
            training_msg = 'training mode' if self.training else 'training disabled'
            win.write_line(1, training_msg)
            win.write_line(2, 'reset statistics: %s' % self.reset_stat)
            win.write_line(3, 'elapsed time: %s, AUTO START: %s' % (repr(time.time()-AUTO_START), repr(AUTO_START)))
            win.write_line(4, 'intention: %s' % (repr(self.intention)))
            win.write_line(5, 'key: %s' % (self.key))
            win.write_line(6, 'latency, %.4f' % (self.latency))
            win.write_line(7, 'max_x: %.4f max_z: %.4f' % (self.MAX_X, self.MAX_Z))
            control_name = self.current_control.name if self.current_control else 'None'
            win.write_line(8, 'control: %s ~ %s' % (control_name, self.log))
            win.refresh()

        self.pub.publish(self.twist)

    def callback(self, data):
	self.tele_twist.linear.x = self.MAX_X*data.axes[JOY_MAPPING['axes']['left_stick_ud']]
	self.tele_twist.angular.z = self.MAX_Z*data.axes[JOY_MAPPING['axes']['left_stick_lr']]
	# reset max_x, max_y
	self.MAX_X = self.MAX_X*(1-data.buttons[JOY_MAPPING['buttons']['left_stick_btn']])
	self.MAX_Z = self.MAX_Z*(1-data.buttons[JOY_MAPPING['buttons']['left_stick_btn']])
	self.MAX_X += data.buttons[JOY_MAPPING['buttons']['lb']]
	self.MAX_Z += data.buttons[JOY_MAPPING['buttons']['rb']]

        # parse control key
        if data.buttons[JOY_MAPPING['buttons']['back']] == 1:
            self.key = 'q'
        if data.buttons[JOY_MAPPING['buttons']['A']] == 1:
            self.key = 'a'
        if data.buttons[JOY_MAPPING['buttons']['Y']] == 1:
            self.key = 'h'
        if data.buttons[JOY_MAPPING['buttons']['B']] == 1:
            self.key = 'rst'
        if data.buttons[JOY_MAPPING['buttons']['start']] == 1:
            self.key = 't'
        if data.axes[JOY_MAPPING['axes']['left_button_lr']] == 1:
            self.key = 'll'
        if data.axes[JOY_MAPPING['axes']['left_button_lr']] == -1:
            self.key = 'rr'
        if data.axes[JOY_MAPPING['axes']['left_button_ud']] == 1:
            self.key = 'ii'
        if data.axes[JOY_MAPPING['axes']['left_button_ud']] == -1:
            self.key = 'kk'

    def run(self):
        while not self.exiting:
            self.control_loop()
            self.rate.sleep()

class AutoControl(object):
    def __init__(self, clf, task, keys='a'):
        self.clf = clf
        self.keys = keys
        self.name = 'Auto Controller %s' % (task)

    def control_loop(self, s):
        if s.reset_stat:
            global AUTO_START
            AUTO_START=time.time()
            self.clf.stat.reset()
            self.clf.stat.logger.info('AutoReset$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$')

        msg = s.image_msg
        if msg:
            ts = s.image_msgs[-1].header.stamp
            imgs = [cvbridge.imgmsg_to_cv2(msg, desired_encoding='bgr8') for msg in list(s.image_msgs)]

            if s.intention == config.STOP:
                twist = Twist()
            else:
                twist = self.clf.predict_twist(imgs, s.intention)
            s.set_twist(twist)
            s.latency = (rospy.get_rostime() - ts).to_sec()
            s.log = '\n %s\n%s' % (repr(twist), self.clf.stat.str())
        else:
            s.log = 'None Image'

        self.clf.stat.log()

class TeleControl(object):
    keys = 'h'

    def __init__(self):
	# publishing as training data
	self.pub_teleop_vel = rospy.Publisher('train/cmd_vel', Twist, queue_size=1)
	self.pub_image = rospy.Publisher('train/image_raw', Image, queue_size=1)
        self.pub_intention = rospy.Publisher('train/intention', String, queue_size=1)
        self.name = "TeleController"
       # used for quatitive results
        self.stat = Stat(self.name)

    def control_loop(self, s):
	s.set_twist(s.tele_twist)

	if s.training:
	    self.publish(s)
        if s.reset_stat:
            self.stat.logger.info('TeleReset$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$')
	    self.stat.reset()

	self.stat.include([s.twist.linear.x, s.twist.angular.z])
	s.log = 'teleop got key %s\n%s\n%s' % (s.key, repr(s.tele_twist), self.stat.str())
        self.stat.log()

    def publish(self, s):
        if s.image_msg:
            print 'hahaahah', s.intention
            self.pub_image.publish(s.image_msg)
            self.pub_teleop_vel.publish(s.twist)
            self.pub_intention.publish(s.intention)

def control(stdscr):
    rospy.init_node('controller')
    win = TextWindow(stdscr)
    con = Controller(win)
    con.register(TeleControl())
    clf = None
    clf = Policy(config.TASK)
    con.register(AutoControl(clf, config.TASK, 'a'))

    try:
        con.run()
    finally:
        con.pub.publish(Twist())
    return clf

def test_policy():
    rospy.init_node('controller')
    con = Controller(None)
    con.register(TeleControl())
    #clf = None
    clf = Policy(config.TASK)
    con.register(AutoControl(clf, config.TASK, 'a'))

    try:
        con.run()
    finally:
        con.pub.publish(Twist())
    return clf

def main():
    try:
        global AUTO_START
	AUTO_START = time.time()
        clf = curses.wrapper(control)
        elapsed = time.time() - AUTO_START
        print(clf.stat.str())
        clf.stat.log()
        clf.stat.logger.info('Time to fulfile the task %f seconds' % elapsed)
        print('Time to finish the task %f seconds' % elapsed)
    except rospy.ROSInterruptException:
        pass

if __name__ == '__main__':
    #test_policy()
    main()
