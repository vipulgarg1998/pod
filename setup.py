import os
from glob import glob
from setuptools import setup

package_name = 'pose_mimic'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        # (os.path.join(package_name, 'utils'), glob(package_name + '/utils/*.py')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='vipul',
    maintainer_email='vipul.garg@ipresence.jp',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'zed_publisher = pose_mimic.zed_publisher:main',
            'pose_estimator = pose_mimic.pose_estimator:main',
        ],
    },
)
