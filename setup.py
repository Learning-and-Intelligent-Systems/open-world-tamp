"""Setup script."""

from setuptools import setup

setup(
    name="owt",
    version="1.0.0",
    packages=["owt"],
    include_package_data=True,
    install_requires=[
        "numpy",
        "scipy",
        "pybullet",
        "trimesh",
        "torch",
        "torchvision",
        "easydict",
        "transforms3d",
        "imageio",
        "opencv-python",
        "rtree",
        "zmq",
        "matplotlib",
        "open3d",
        "plyfile",
        "prefetch_generator",
        "transforms3d",
        "SpeechRecognition",
        "scikit-learn",
        "openai",
        "pyyaml",
        "opencv-python",
        "SpeechRecognition",
        "pytest",
        "pytest-cov",
    ],
)
