from glob import glob
import os

from setuptools import find_packages, setup

package_name = 'my_ekf_pkg'

setup(
    name=package_name,
    version='0.1.0',
    packages=find_packages(),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.launch.py')),
        (os.path.join('share', package_name, 'config'), glob('config/*.yaml')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='noahkiw',
    maintainer_email='noahkiw@todo.todo',
    description='Own EKF (unicycle model, odometry + IMU + 1 Hz ground truth) and '
                'robot_localization comparison for IST Intro to Robotics Mini-Project 1, '
                'Task 1a, with evaluation tools.',
    license='TODO: License declaration',
    entry_points={
        'console_scripts': [
            'ekf_node = my_ekf_pkg.ekf_node:main',
            'gt_publisher = my_ekf_pkg.gt_publisher:main',
            'evaluate = my_ekf_pkg.evaluate:main',
            'run_offline = my_ekf_pkg.run_offline:main',
            'plot_results = my_ekf_pkg.plot_results:main',
        ],
    },
)
