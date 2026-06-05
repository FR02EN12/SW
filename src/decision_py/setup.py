from setuptools import find_packages, setup

package_name = 'decision_py'

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
    description='Python decision nodes for FSD cooperative driving',
    license='Apache-2.0',
    entry_points={
        'console_scripts': [
            'drive_mode = decision_py.drive_mode:main',
            'cooperation = decision_py.cooperation:main',
            'behavior_plan = decision_py.behavior_plan:main',
        ],
    },
)
