from launch import LaunchDescription
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
from pathlib import Path


def generate_launch_description():
    share_dir = Path(get_package_share_directory("go2_mimickit_deploy"))
    params = share_dir / "config" / "go2_policy_params.yaml"
    return LaunchDescription(
        [
            Node(
                package="go2_mimickit_deploy",
                executable="go2_policy_node",
                name="go2_policy_node",
                output="screen",
                parameters=[str(params)],
            )
        ]
    )
