#include <cv_bridge/cv_bridge.h>
#include <octomap/OcTree.h>
#include <octomap_msgs/conversions.h>
#include <cbf_circ_interfaces/srv/find_frontier_points.hpp>
#include <cbf_circ_interfaces/srv/find_neighbor_points.hpp>
#include <image_transport/image_transport.hpp>
#include <memory>
#include <mutex>
#include <octomap_msgs/msg/octomap.hpp>
#include <opencv2/imgproc.hpp>
#include <random>
#include <rclcpp/rclcpp.hpp>

class OctomapWithQuery : public rclcpp::Node {
 public:
  OctomapWithQuery()
      : Node("octomap_with_query_node"),
        node_handle_(std::shared_ptr<OctomapWithQuery>(this, [](auto*) {})),
        image_transport_(node_handle_) {
    // Image transport
    debug_visualizer = image_transport_.advertise("frontier_debug", 1);

    // Octomap subscriber
    octomap_subscriber_ = this->create_subscription<octomap_msgs::msg::Octomap>(
        "octomap_binary", 1,
        std::bind(&OctomapWithQuery::OctomapCallback, this, std::placeholders::_1));

    // Services
    neighbor_service_ =
        this->create_service<cbf_circ_interfaces::srv::FindNeighborPoints>(
            "neighbor_points",
            std::bind(&OctomapWithQuery::QueryNeighboringPoints, this,
                      std::placeholders::_1, std::placeholders::_2));
    frontier_service_ =
        this->create_service<cbf_circ_interfaces::srv::FindFrontierPoints>(
            "frontier_points", std::bind(&OctomapWithQuery::QueryFrontierPoints, this,
                                         std::placeholders::_1, std::placeholders::_2));
  }

  // Stores contour information for both internal and external contours
  struct Contour {
    std::vector<cv::Point> external;               // Points at the outermost boundary
    std::vector<std::vector<cv::Point>> internal;  // Points at the boundary of holes
    bool valid;
  };

 private:
  // Storage and mutex lock for the latest map received
  std::unique_ptr<octomap::OcTree> octomap_ptr;
  std::mutex octomap_lock;

  // Image transport for debug
  rclcpp::Node::SharedPtr node_handle_;
  image_transport::ImageTransport image_transport_;
  image_transport::Publisher debug_visualizer;  // Debug visualization

  // Subscribers & services
  rclcpp::Subscription<octomap_msgs::msg::Octomap>::SharedPtr octomap_subscriber_;
  rclcpp::Service<cbf_circ_interfaces::srv::FindNeighborPoints>::SharedPtr
      neighbor_service_;
  rclcpp::Service<cbf_circ_interfaces::srv::FindFrontierPoints>::SharedPtr
      frontier_service_;

  // Algorithm to extract the contours (inner & outer) of a given region
  Contour ExtractContour(const cv::Mat& free, const cv::Point& origin) {
    std::vector<std::vector<cv::Point>> contours;
    std::vector<cv::Vec4i> hierarchy;
    cv::findContours(free, contours, hierarchy, cv::RETR_TREE, cv::CHAIN_APPROX_NONE);

    // Select the contour containing the origin
    Contour relevant_contour;
    relevant_contour.valid = false;

    for (size_t i = 0; i < contours.size(); ++i) {
      bool origin_in_hole = false;

      const std::vector<cv::Point>& contour = contours[i];

      // Check if the origin is inside the outermost boundaries
      if (cv::pointPolygonTest(contour, origin, false) > 0) {
        // Check if the origin is not inside the holes (children) of the contour
        std::vector<std::vector<cv::Point>> children_contours;
        for (size_t j = 0; j < contours.size(); ++j)
          if (hierarchy[j][3] == static_cast<int>(i))  // Parent is the current contour
            children_contours.push_back(contours[j]);

        for (const std::vector<cv::Point>& child_contour : children_contours) {
          // If the origin is inside a hole, then this is the incorrect contour
          if (cv::pointPolygonTest(child_contour, origin, false) > 0)
            origin_in_hole = true;
          break;
        }

        // If the origin is not in any of the holes, then the current contour is
        // accurate
        if (!origin_in_hole) {
          relevant_contour.external = contour;
          relevant_contour.internal = children_contours;
          relevant_contour.valid = true;
        }
      }
    }

    if (!relevant_contour.valid) {
      RCLCPP_ERROR(this->get_logger(), "Contour cannot be set properly. Investigate");
    }

    return relevant_contour;
  }

  double MilliSecondsSinceTime(const rclcpp::Time& start) {
    return (this->now() - start).nanoseconds() * 1e-6;
  }

  void OctomapCallback(const octomap_msgs::msg::Octomap::SharedPtr msg) {
    // Reinitialize tree
    {
      std::unique_lock<std::mutex> lock(octomap_lock);
      // const double kResolution = msg.resolution;
      octomap_ptr.reset(
          dynamic_cast<octomap::OcTree*>(octomap_msgs::binaryMsgToMap(*msg)));
      octomap_ptr->expand();
    }
  }

  void QueryNeighboringPoints(
      const std::shared_ptr<cbf_circ_interfaces::srv::FindNeighborPoints::Request>
          request,
      std::shared_ptr<cbf_circ_interfaces::srv::FindNeighborPoints::Response>
          response) {
    rclcpp::Time start = this->now();
    RCLCPP_DEBUG(this->get_logger(), "Neighbor query received");

    // Cannot proceed without a map
    if (octomap_ptr == nullptr) {
      RCLCPP_WARN(this->get_logger(), "Map is not initialized");
      return;
    }

    geometry_msgs::msg::Point query_pt = request->query;

    // Set bounds of the map to be extracted
    {
      std::unique_lock<std::mutex> lock(octomap_lock);
      octomap::point3d max_bounds(query_pt.x + request->radius,
                                  query_pt.y + request->radius,
                                  query_pt.z + request->radius),
          min_bounds(query_pt.x - request->radius, query_pt.y - request->radius,
                     query_pt.z - request->radius);

      for (octomap::OcTree::leaf_bbx_iterator
               it = octomap_ptr->begin_leafs_bbx(min_bounds, max_bounds),
               end = octomap_ptr->end_leafs_bbx();
           it != end; ++it) {
        // If logOdd > 0 -> Occupied. Otherwise free
        if (it->getLogOdds() > 0) {
          double x = it.getX(), y = it.getY(), z = it.getZ();
          geometry_msgs::msg::Point pt;
          pt.x = x;
          pt.y = y;
          pt.z = z;
          response->neighbors.push_back(pt);
        }
      }
    }

    double execution_time = MilliSecondsSinceTime(start);
    RCLCPP_INFO_STREAM(
        this->get_logger(),
        "Found " << response->neighbors.size() << " neighbors for point (" << query_pt.x
                 << ", " << query_pt.y << ", " << query_pt.z << ") at radius "
                 << request->radius << " m in " << execution_time << "ms");

    return;
  }

  void VisualizeFrontierCall(
      const cv::Mat& free_map,
      const cv::Mat& occupied_map,
      const std::shared_ptr<cbf_circ_interfaces::srv::FindFrontierPoints::Response>
          frontier) {
    cv::Mat visual;
    cv::Mat zero_image(free_map.rows, free_map.cols, CV_8UC1, cv::Scalar(0));
    std::vector<cv::Mat> channels{zero_image, free_map, occupied_map};
    cv::merge(channels, visual);

    // Add different color for each cluster
    static std::random_device dev;
    static std::mt19937 rng(dev());
    static std::uniform_int_distribution<std::mt19937::result_type> sample_255(0, 255);
    cv::Scalar cluster_color(255, 0, 255);
    size_t previous_cluster_id = 0;

    for (size_t i = 0; i < frontier->frontiers.size(); ++i) {
      // Resample color for a new cluster
      if (previous_cluster_id != frontier->cluster_id[i]) {
        cluster_color = cv::Scalar(sample_255(rng), sample_255(rng), sample_255(rng));
      }

      cv::Point pt(frontier->frontiers[i].x, frontier->frontiers[i].y);
      cv::circle(visual, pt, 0, cluster_color, 1);

      previous_cluster_id = frontier->cluster_id[i];
    }

    // Correct the direction for better visualization
    cv::flip(visual, visual, 0);
    sensor_msgs::msg::Image::SharedPtr visMsg =
        cv_bridge::CvImage(std_msgs::msg::Header(), "bgr8", visual).toImageMsg();
    debug_visualizer.publish(visMsg);
  }

  void FindFrontierPoints(
      const cv::Mat& free_map,
      const cv::Mat& occupied_map,
      const cv::Point& map_origin,
      std::shared_ptr<cbf_circ_interfaces::srv::FindFrontierPoints::Response>
          response) {
    Contour contour = ExtractContour(free_map, map_origin);

    // All external and internal points are at the boundary & are therefore valid
    // frontiers. Flatten the found contour
    std::vector<cv::Point> boundary_points;
    boundary_points.insert(boundary_points.end(), contour.external.begin(),
                           contour.external.end());
    for (const std::vector<cv::Point>& internal_contour : contour.internal)
      boundary_points.insert(boundary_points.end(), internal_contour.begin(),
                             internal_contour.end());

    if (boundary_points.size() == 0) {
      RCLCPP_WARN(this->get_logger(), "No frontier points found");
      return;
    }

    // Frontier points are the contours of free  space that are not adjacent to an
    // occupied location. Dilate the occupied cells with 3x3 kernel to extend the region
    // by 1 pixel. Alternative to querying 8-neighborhood for every contour
    cv::Mat kernel =
        cv::getStructuringElement(cv::MorphShapes::MORPH_RECT, cv::Size(3, 3));
    cv::Mat occupied_contour;
    cv::dilate(occupied_map, occupied_contour, kernel);

    // Discontinuity indicates a cluster change for frontiers. Identified by a max.
    // coordinate distance of greater than 1
    cv::Point previous_pt = boundary_points[0];
    size_t cluster_id = 0;
    for (const auto& pt : boundary_points) {
      if (occupied_contour.at<uchar>(pt) != 255) {
        geometry_msgs::msg::Point coordinate;
        coordinate.x = pt.x;
        coordinate.y = pt.y;
        response->frontiers.push_back(coordinate);

        // Increment cluster id if the frontier point is discontinuous
        // FIXME: If the cluster begins in the middle of a frontier line, the cluster
        // will be treated as two different clusters. Can be post-processed
        size_t distance =
            std::max(std::abs(previous_pt.x - pt.x), std::abs(previous_pt.y - pt.y));
        if (distance > 1)
          ++cluster_id;

        response->cluster_id.push_back(cluster_id);
        previous_pt = pt;
      }
    }
    return;
  }

  bool QueryFrontierPoints(
      const std::shared_ptr<cbf_circ_interfaces::srv::FindFrontierPoints::Request>
          request,
      std::shared_ptr<cbf_circ_interfaces::srv::FindFrontierPoints::Response>
          response) {
    rclcpp::Time start = this->now();
    RCLCPP_DEBUG(this->get_logger(), "Frontier query received");

    // Cannot proceed without a map
    if (octomap_ptr == nullptr) {
      RCLCPP_WARN(this->get_logger(), "Map is not initialized");
      return false;
    }

    // // Copy map (expensive ~ 7ms)
    // section = ros::WallTime::now();
    // std::unique_ptr<octomap::OcTree> octomap_ptr_local;
    // {
    //   std::unique_lock<std::mutex> lock(octomap_lock);
    //   octomap_ptr_local =
    //       std::make_unique<octomap::OcTree>(octomap::OcTree(*octomap_ptr));
    // }
    // double section_time = MilliSecondsSinceTime(section);
    // ROS_INFO_STREAM("Map copy completed in " << section_time << " ms");

    // Pixelize, with 2px buffer for boundary points
    double x_max, y_max, z_max, x_min, y_min, z_min;
    octomap_ptr->getMetricMax(x_max, y_max, z_max);
    octomap_ptr->getMetricMin(x_min, y_min, z_min);

    const double map_resolution = octomap_ptr->getResolution(), buffer_factor = 2.0;

    x_max += map_resolution * buffer_factor;
    y_max += map_resolution * buffer_factor;
    x_min -= map_resolution * buffer_factor;
    y_min -= map_resolution * buffer_factor;

    const size_t image_width =
                     static_cast<size_t>(std::ceil((x_max - x_min) / map_resolution)),
                 image_height =
                     static_cast<size_t>(std::ceil((y_max - y_min) / map_resolution));

    const size_t map_origin_x = static_cast<size_t>(std::ceil(-x_min / map_resolution)),
                 map_origin_y = static_cast<size_t>(std::ceil(-y_min / map_resolution));
    const cv::Point map_origin(map_origin_x, map_origin_y);

    octomap::point3d max_bounds(x_max, y_max, request->z_max),
        min_bounds(x_min, y_min, request->z_min);
    cv::Mat occupied_map(image_height, image_width, CV_8UC1, cv::Scalar(0)),
        free_map(image_height, image_width, CV_8UC1, cv::Scalar(0));

    // Mutex access to the tree
    {
      std::unique_lock<std::mutex> lock(octomap_lock);
      rclcpp::Time section = this->now();
      for (octomap::OcTree::leaf_bbx_iterator
               it = octomap_ptr->begin_leafs_bbx(min_bounds, max_bounds),
               end = octomap_ptr->end_leafs_bbx();
           it != end; ++it) {
        size_t x_coord =
                   static_cast<size_t>(std::ceil((it.getX() - x_min) / map_resolution)),
               y_coord =
                   static_cast<size_t>(std::ceil((it.getY() - y_min) / map_resolution));

        // If logOdd > 0 -> Occupied. Otherwise free
        // Checks for overlapping free / occupied is not essential
        if (it->getLogOdds() > 0) {
          occupied_map.at<uint8_t>(y_coord, x_coord) = 255;
          free_map.at<uint8_t>(y_coord, x_coord) = 0;
        } else {
          if (occupied_map.at<uint8_t>(y_coord, x_coord) == 0)
            free_map.at<uint8_t>(y_coord, x_coord) = 255;
        }
      }
      RCLCPP_DEBUG_STREAM(this->get_logger(), "Occupied / free projection completed in "
                                                  << MilliSecondsSinceTime(section)
                                                  << " ms");
    }

    // Frontier point extraction
    {
      rclcpp::Time section = this->now();
      FindFrontierPoints(free_map, occupied_map, map_origin, response);
      RCLCPP_DEBUG_STREAM(this->get_logger(), "Frontier idenfitication completed in "
                                                  << MilliSecondsSinceTime(section)
                                                  << " ms");
    }

    // DEBUG Visualize before post-processing
    VisualizeFrontierCall(free_map, occupied_map, response);

    // Post-processing to convert points into real coordinates
    for (size_t i = 0; i < response->frontiers.size(); ++i) {
      response->frontiers[i].x = response->frontiers[i].x * map_resolution + x_min;
      response->frontiers[i].y = response->frontiers[i].y * map_resolution + y_min;
    }

    RCLCPP_INFO_STREAM(this->get_logger(),
                       "Found " << response->frontiers.size()
                                << " frontiers for map projection between heights ("
                                << request->z_min << ", " << request->z_max << ") in "
                                << MilliSecondsSinceTime(start) << "ms");

    return true;
  }
};

int main(int argc, char** argv) {
  rclcpp::init(argc, argv);
  std::shared_ptr<OctomapWithQuery> node = std::make_shared<OctomapWithQuery>();
  rclcpp::spin(node);
  rclcpp::shutdown();
}
