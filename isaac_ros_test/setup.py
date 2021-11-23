from setuptools import setup

package_name = 'isaac_ros_test'

setup(
    name=package_name,
    version='0.9.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Hemal Shah',
    maintainer_email='hemals@nvidia.com',
    description='Isaac ROS testing utilities',
    license='NVIDIA Isaac ROS Software License',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
        ],
    },
)
