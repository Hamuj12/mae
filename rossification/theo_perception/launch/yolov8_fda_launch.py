from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description() -> LaunchDescription:
    namespace_arg = DeclareLaunchArgument('namespace', default_value='eowyn')
    model_path_arg = DeclareLaunchArgument('model_path', default_value='')
    topic_in_arg = DeclareLaunchArgument('topic_in', default_value='/camera/color/image_raw')

    config_file = PathJoinSubstitution(
        [FindPackageShare('theo_perception'), 'config', 'yolov8_fda_node.yaml']
    )

    node = Node(
        package='theo_perception',
        executable='yolov8_fda_node',
        name='yolov8_fda_node',
        namespace=LaunchConfiguration('namespace'),
        parameters=[
            config_file,
            {
                'model_path': LaunchConfiguration('model_path'),
                'topic_in': LaunchConfiguration('topic_in'),
            },
        ],
        output='screen',
    )

    return LaunchDescription([namespace_arg, model_path_arg, topic_in_arg, node])
