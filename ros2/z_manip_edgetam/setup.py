from glob import glob

from setuptools import find_packages, setup


package_name = 'z_manip_edgetam'

setup(
    name=package_name,
    version='0.1.0',
    packages=find_packages(exclude=('test',)),
    data_files=[
        ('share/ament_index/resource_index/packages', [f'resource/{package_name}']),
        (f'share/{package_name}', ['package.xml', 'README.md']),
        (f'share/{package_name}/config', glob('config/*.yaml')),
        (f'share/{package_name}/launch', glob('launch/*.launch.py')),
    ],
    install_requires=['setuptools', 'numpy'],
    zip_safe=True,
    maintainer='Z Robotics Lab',
    maintainer_email='robotics@invalid.local',
    description='Fail-closed RGB-D ROS 2 adapter for an external EdgeTAM service',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'edgetam_adapter = z_manip_edgetam.node:main',
        ],
    },
)
