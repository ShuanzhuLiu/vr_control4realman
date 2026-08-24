from setuptools import find_packages, setup


package_name = "vr_data_pub"


setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        (
            "share/ament_index/resource_index/packages",
            ["resource/" + package_name],
        ),
        ("share/" + package_name, ["package.xml"]),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="lh",
    maintainer_email="lh@example.com",
    description="PICO TCP pose receiver and ROS 2 topic distributor",
    license="Apache-2.0",
    entry_points={
        "console_scripts": [
            "vr_data_pub = vr_data_pub.vr_data_pub_long_connect:main",
            "vr_data_distributer = vr_data_pub.vr_data_distributer:main",
        ],
    },
)
