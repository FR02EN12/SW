from glob import glob
import os

from setuptools import find_packages, setup

package_name = 'perception_py'

setup(
    name=package_name,
    version='1.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'config'), glob('config/*')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='jhp',
    maintainer_email='jhp@todo.todo',
    description='Python perception nodes for lane, LED, space, and situation perception',
    license='Apache-2.0',
    entry_points={
        'console_scripts': [
            'lane_final = perception_py.lane_final:main',
            'camera_driver = perception_py.camera_driver:main',
            'led_signal = perception_py.led_signal:main',
            'space_memory = perception_py.space_memory:main',
            'situation_fusion = perception_py.situation_fusion:main',
        ],
    },
)
