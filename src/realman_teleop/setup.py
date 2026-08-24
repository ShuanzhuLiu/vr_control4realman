from glob import glob

from setuptools import find_packages, setup


package_name = "realman_teleop"


setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/realman_teleop"]),
        (f"share/{package_name}", ["package.xml", "README.md"]),
        (f"share/{package_name}/launch", glob("launch/*.launch.py")),
    ],
    install_requires=["setuptools"],
    zip_safe=False,
    maintainer="lh",
    maintainer_email="lh@example.com",
    description="Safe ROS 2 bridge from TAMEn Cartesian commands to RealMan dual arms",
    license="Apache-2.0",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            "realman_dual_arm_bridge = realman_teleop.bridge:main",
            "realman_pose_converter = realman_teleop.pose_converter:main",
        ],
    },
)
