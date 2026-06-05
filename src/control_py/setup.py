from setuptools import find_packages, setup

package_name = 'control_py'

setup(
    name=package_name,
    version='1.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='jhp',
    maintainer_email='jhp@todo.todo',
    description='Python control nodes for lane following and LED output',
    license='Apache-2.0',
    entry_points={
        'console_scripts': [
            'lane_follow = control_py.lane_follow:main',
            'led_behavior = control_py.led_behavior:main',
            'led_io = control_py.led_io:main',
        ],
    },
)
