from glob import glob
from setuptools import find_packages, setup


package_name = "z_manip_motion"

setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(exclude=("test",)),
    data_files=[
        ("share/ament_index/resource_index/packages", [f"resource/{package_name}"]),
        (f"share/{package_name}", ["package.xml", "README.md"]),
        (f"share/{package_name}/config", glob("config/*.yaml") + glob("config/*.srdf")),
        (f"share/{package_name}/launch", glob("launch/*.launch.py")),
    ],
    install_requires=["setuptools"],
    tests_require=["pytest"],
    zip_safe=True,
    maintainer="Z Robotics Lab",
    maintainer_email="robotics@z-robotics.local",
    description="Fail-closed MoveIt 2 planning bridge for PiPER.",
    license="Apache-2.0",
    entry_points={
        "console_scripts": [
            "complete_joint_state = z_manip_motion.joint_state_assembler:main",
            "motion_plan_bridge = z_manip_motion.moveit_plan_bridge:main",
        ],
    },
)
