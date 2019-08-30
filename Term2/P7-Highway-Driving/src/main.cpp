#include <fstream>
#include <math.h>
#include <uWS/uWS.h>
#include <chrono>
#include <iostream>
#include <thread>
#include <vector>
#include "Eigen-3.3/Eigen/Core"
#include "Eigen-3.3/Eigen/QR"
#include "json.hpp"
#include "spline.h"

using namespace std;

// for convenience
using json = nlohmann::json;

// For converting back and forth between radians and degrees.
constexpr double pi() { return M_PI; }
double deg2rad(double x) { return x * pi() / 180; }
double rad2deg(double x) { return x * 180 / pi(); }

// Checks if the SocketIO event has JSON data.
// If there is data the JSON object in string format will be returned,
// else the empty string "" will be returned.
string hasData(string s) {
  auto found_null = s.find("null");
  auto b1 = s.find_first_of("[");
  auto b2 = s.find_first_of("}");
  if (found_null != string::npos) {
    return "";
  } else if (b1 != string::npos && b2 != string::npos) {
    return s.substr(b1, b2 - b1 + 2);
  }
  return "";
}

double distance(double x1, double y1, double x2, double y2)
{
	return sqrt((x2-x1)*(x2-x1)+(y2-y1)*(y2-y1));
}
int ClosestWaypoint(double x, double y, const vector<double> &maps_x, const vector<double> &maps_y)
{

	double closestLen = 100000; //large number
	int closestWaypoint = 0;

	for(int i = 0; i < maps_x.size(); i++)
	{
		double map_x = maps_x[i];
		double map_y = maps_y[i];
		double dist = distance(x,y,map_x,map_y);
		if(dist < closestLen)
		{
			closestLen = dist;
			closestWaypoint = i;
		}

	}

	return closestWaypoint;

}

int NextWaypoint(double x, double y, double theta, const vector<double> &maps_x, const vector<double> &maps_y)
{

	int closestWaypoint = ClosestWaypoint(x,y,maps_x,maps_y);

	double map_x = maps_x[closestWaypoint];
	double map_y = maps_y[closestWaypoint];

	double heading = atan2( (map_y-y),(map_x-x) );

	double angle = abs(theta-heading);

	if(angle > pi()/4)
	{
		closestWaypoint++;
	}

	return closestWaypoint;

}

// Transform from Cartesian x,y coordinates to Frenet s,d coordinates
vector<double> getFrenet(double x, double y, double theta, const vector<double> &maps_x, const vector<double> &maps_y)
{
	int next_wp = NextWaypoint(x,y, theta, maps_x,maps_y);

	int prev_wp;
	prev_wp = next_wp-1;
	if(next_wp == 0)
	{
		prev_wp  = maps_x.size()-1;
	}

	double n_x = maps_x[next_wp]-maps_x[prev_wp];
	double n_y = maps_y[next_wp]-maps_y[prev_wp];
	double x_x = x - maps_x[prev_wp];
	double x_y = y - maps_y[prev_wp];

	// find the projection of x onto n
	double proj_norm = (x_x*n_x+x_y*n_y)/(n_x*n_x+n_y*n_y);
	double proj_x = proj_norm*n_x;
	double proj_y = proj_norm*n_y;

	double frenet_d = distance(x_x,x_y,proj_x,proj_y);

	//see if d value is positive or negative by comparing it to a center point

	double center_x = 1000-maps_x[prev_wp];
	double center_y = 2000-maps_y[prev_wp];
	double centerToPos = distance(center_x,center_y,x_x,x_y);
	double centerToRef = distance(center_x,center_y,proj_x,proj_y);

	if(centerToPos <= centerToRef)
	{
		frenet_d *= -1;
	}

	// calculate s value
	double frenet_s = 0;
	for(int i = 0; i < prev_wp; i++)
	{
		frenet_s += distance(maps_x[i],maps_y[i],maps_x[i+1],maps_y[i+1]);
	}

	frenet_s += distance(0,0,proj_x,proj_y);

	return {frenet_s,frenet_d};

}

// Transform from Frenet s,d coordinates to Cartesian x,y
vector<double> getXY(double s, double d, const vector<double> &maps_s, const vector<double> &maps_x, const vector<double> &maps_y)
{
	int prev_wp = -1;

	while(s > maps_s[prev_wp+1] && (prev_wp < (int)(maps_s.size()-1) ))
	{
		prev_wp++;
	}

	int wp2 = (prev_wp+1)%maps_x.size();

	double heading = atan2((maps_y[wp2]-maps_y[prev_wp]),(maps_x[wp2]-maps_x[prev_wp]));
	// the x,y,s along the segment
	double seg_s = (s-maps_s[prev_wp]);

	double seg_x = maps_x[prev_wp]+seg_s*cos(heading);
	double seg_y = maps_y[prev_wp]+seg_s*sin(heading);

	double perp_heading = heading-pi()/2;

	double x = seg_x + d*cos(perp_heading);
	double y = seg_y + d*sin(perp_heading);

	return {x,y};

}

int main() {
  uWS::Hub h;

  // Load up map values for waypoint's x,y,s and d normalized normal vectors
  vector<double> map_waypoints_x;
  vector<double> map_waypoints_y;
  vector<double> map_waypoints_s;
  vector<double> map_waypoints_dx;
  vector<double> map_waypoints_dy;

  // Waypoint map to read from
  string map_file_ = "../data/highway_map.csv";
  // The max s value before wrapping around the track back to 0
  double max_s = 6945.554;

  ifstream in_map_(map_file_.c_str(), ifstream::in);

  string line;
  while (getline(in_map_, line)) {
  	istringstream iss(line);
  	double x;
  	double y;
  	float s;
  	float d_x;
  	float d_y;
  	iss >> x;
  	iss >> y;
  	iss >> s;
  	iss >> d_x;
  	iss >> d_y;
  	map_waypoints_x.push_back(x);
  	map_waypoints_y.push_back(y);
  	map_waypoints_s.push_back(s);
  	map_waypoints_dx.push_back(d_x);
  	map_waypoints_dy.push_back(d_y);
  }

  // Car's lane. Starting at middle lane.
  int lane = 1;

  // Reference velocity.
  double ref_vel = 0.0; // mph

  h.onMessage([&ref_vel, &lane, &map_waypoints_x,&map_waypoints_y,&map_waypoints_s,&map_waypoints_dx,&map_waypoints_dy]
    (uWS::WebSocket<uWS::SERVER> ws, char *data, size_t length, uWS::OpCode opCode) {



    // "42" at the start of the message means there's a websocket message event.
    // The 4 signifies a websocket message
    // The 2 signifies a websocket event
    //auto sdata = string(data).substr(0, length);
    //cout << sdata << endl;
    if (length && length > 2 && data[0] == '4' && data[1] == '2') {

      auto s = hasData(data);

      if (s != "") {
        auto j = json::parse(s);

        string event = j[0].get<string>();

        if (event == "telemetry") {
          // j[1] is the data JSON object

        	// Main car's localization Data
          	double car_x = j[1]["x"];
          	double car_y = j[1]["y"];
          	double car_s = j[1]["s"];
          	double car_d = j[1]["d"];
          	double car_yaw = j[1]["yaw"];
          	double car_speed = j[1]["speed"];

          	// Previous path data given to the Planner
          	auto previous_path_x = j[1]["previous_path_x"];
          	auto previous_path_y = j[1]["previous_path_y"];
          	// Previous path's end s and d values
          	double end_path_s = j[1]["end_path_s"];
          	double end_path_d = j[1]["end_path_d"];

          	// Sensor Fusion Data, a list of all other cars on the same side of the road.
          	auto sensor_fusion = j[1]["sensor_fusion"];

            // Provided previous path point size.
            int prev_size = previous_path_x.size();

            if (prev_size > 0)
            {
              car_s = end_path_s;
            }

            // check other car's positions.
            bool car_ahead = false;
            bool car_left = false;
            bool car_right = false;

            double front_threshold = 30;
            double back_threshold = 10;

            for ( int i = 0; i < sensor_fusion.size(); i++ )
            {
                // check other car's speed and position
                double other_vx = sensor_fusion[i][3];
                double other_vy = sensor_fusion[i][4];
                double check_speed = sqrt(other_vx*other_vx + other_vy*other_vy);
                double other_car_s = sensor_fusion[i][5];

                // estimate car's position after executing previous trajectory.
                other_car_s += ((double)prev_size * 0.02 * check_speed);


                float other_car_d = sensor_fusion[i][6];

                // lane is 4m wide
                int other_car_lane = floor(other_car_d / 4.0);

                // check if the other car is on the same lane we are
                // car is in our lane
                if ( other_car_lane == lane )
                {
                    if (other_car_s > car_s && other_car_s - car_s < front_threshold)
                    {
                        car_ahead = true;
                    }

                }
                // car is in left lane
                else if ( other_car_lane - lane == -1 )
                {
                    if (car_s - other_car_s < back_threshold && other_car_s - car_s < front_threshold)
                    {
                        car_left = true;
                    }
                }
                // car is in right lane
                else if ( other_car_lane - lane == 1 )
                {
                    if (car_s - other_car_s < back_threshold && other_car_s - car_s < front_threshold)
                    {
                        car_right = true;
                    }
                }
            }

            // lane-change action if there is a car ahead
            const double max_speed = 49.5; //mph
            const double acc = 0.224; //mph. equals 0.1m/s
            if (car_ahead)
            {
                if (lane > 0 && car_left == false)
                {
                    lane--;
                }
                else if (lane < 2 && car_right == false)
                {
                    lane++;
                }
                else
                {
                    ref_vel = ref_vel - acc;
                }

            }
            else
            {
                // check if car is in the middle lane, if not, change it back
                if ( lane != 1 )
                {
                    if ((lane == 2 && car_left == false) || (lane == 0 && car_right == false))
                    {
                        lane = 1;
                    }
                }
                if ( ref_vel < max_speed )
                {
                    ref_vel = ref_vel + acc;
                }
                if ( ref_vel > max_speed )
                {
                    ref_vel = max_speed;
                }
            }

            // create a list of widely spaced (x,y) as anchor points, evenly spaced at 30m
            // later we will interpolate these waypoints with a spline and fill it in with more points that control speed
            vector<double> anchor_pts_x;
            vector<double> anchor_pts_y;

            // reference x, y, yaw state
            // either be the car location or the last point of the previous path
            double ref_x = car_x;
            double ref_y = car_y;
            double ref_yaw = deg2rad(car_yaw);

            // Do I have have previous points
            // if previous size is almost empty, use the car as the staring reference
            if ( prev_size < 2 )
            {
                // use two points that make the path tangent to the car
                double prev_car_x = car_x - cos(car_yaw);
                double prev_car_y = car_y - sin(car_yaw);

                anchor_pts_x.push_back(prev_car_x);
                anchor_pts_x.push_back(car_x);

                anchor_pts_y.push_back(prev_car_y);
                anchor_pts_y.push_back(car_y);
            }
            else
            {
                // Use the last two points from previous path
                ref_x = previous_path_x[prev_size - 1];
                ref_y = previous_path_y[prev_size - 1];

                double ref_x_prev = previous_path_x[prev_size - 2];
                double ref_y_prev = previous_path_y[prev_size - 2];
                ref_yaw = atan2(ref_y - ref_y_prev, ref_x - ref_x_prev);

                anchor_pts_x.push_back(ref_x_prev);
                anchor_pts_x.push_back(ref_x);

                anchor_pts_y.push_back(ref_y_prev);
                anchor_pts_y.push_back(ref_y);
            }

            // In Frenet add evenly 30m spaced points ahead of the starting reference
            vector<double> next_wp0 = getXY(car_s + 30, 2 + 4*lane, map_waypoints_s, map_waypoints_x, map_waypoints_y);
            vector<double> next_wp1 = getXY(car_s + 60, 2 + 4*lane, map_waypoints_s, map_waypoints_x, map_waypoints_y);
            vector<double> next_wp2 = getXY(car_s + 90, 2 + 4*lane, map_waypoints_s, map_waypoints_x, map_waypoints_y);

            anchor_pts_x.push_back(next_wp0[0]);
            anchor_pts_x.push_back(next_wp1[0]);
            anchor_pts_x.push_back(next_wp2[0]);

            anchor_pts_y.push_back(next_wp0[1]);
            anchor_pts_y.push_back(next_wp1[1]);
            anchor_pts_y.push_back(next_wp2[1]);
            
            // until here the anchor_pts has five points, the car location, one previous location and 30m, 60m and 90m ahead locations

            // transform these coordinates into local car coordinates
            // the start point is (0,0)
            // the angle will be 0
            for ( int i = 0; i < anchor_pts_x.size(); i++ )
            {
              //shift car reference angle to 0 degrees

              double shift_x = anchor_pts_x[i] - ref_x;
              double shift_y = anchor_pts_y[i] - ref_y;

              anchor_pts_x[i] = shift_x * cos(0 - ref_yaw) - shift_y * sin(0 - ref_yaw);
              anchor_pts_y[i] = shift_x * sin(0 - ref_yaw) + shift_y * cos(0 - ref_yaw);
            }

            // create the spline
            tk::spline s;

            // set anchor points to the spline
            s.set_points(anchor_pts_x, anchor_pts_y);

            // define actual (x,y) points we will use for the planner
          	vector<double> next_x_vals;
          	vector<double> next_y_vals;

            // add points from previous path - for continuity
            for ( int i = 0; i < prev_size; i++ )
            {
              next_x_vals.push_back(previous_path_x[i]);
              next_y_vals.push_back(previous_path_y[i]);
            }

            // Calculate how to break up spline points so that we travel at our desired reference velocity
            double target_x = 30.0;
            double target_y = s(target_x);
            double target_dist = sqrt((target_x) * (target_x) + (target_y) * (target_y));

            double x_add_on = 0;

            // fill up the rest of our path planner after filling it with previous points, here we will always output 50 points
            for( int i = 1; i < 50 - prev_size; i++ )
            {
                // dividing by 2.24 converts m/s to mile/h
                // calculate spacing of number of points based on car speed
                double N = (target_dist / (0.02 * ref_vel / 2.24));
                double x_point = x_add_on + (target_x) / N;
                double y_point = s(x_point);

                x_add_on = x_point;

                // Convert coordinates from local car coordinates to normal

                double x_ref = x_point;
                double y_ref = y_point;

                // rotate back to normal after rotating it earlier
                x_point = (x_ref * cos(ref_yaw)-y_ref*sin(ref_yaw));
                y_point = (x_ref * sin(ref_yaw)+y_ref*cos(ref_yaw));

                x_point += ref_x;
                y_point += ref_y;

                next_x_vals.push_back(x_point);
                next_y_vals.push_back(y_point);

            }

            json msgJson;

          	msgJson["next_x"] = next_x_vals;
          	msgJson["next_y"] = next_y_vals;

          	auto msg = "42[\"control\","+ msgJson.dump()+"]";

          	//this_thread::sleep_for(chrono::milliseconds(1000));
          	ws.send(msg.data(), msg.length(), uWS::OpCode::TEXT);

        }
      } else {
        // Manual driving
        std::string msg = "42[\"manual\",{}]";
        ws.send(msg.data(), msg.length(), uWS::OpCode::TEXT);
      }
    }
  });

  // We don't need this since we're not using HTTP but if it's removed the
  // program
  // doesn't compile :-(
  h.onHttpRequest([](uWS::HttpResponse *res, uWS::HttpRequest req, char *data,
                     size_t, size_t) {
    const std::string s = "<h1>Hello world!</h1>";
    if (req.getUrl().valueLength == 1) {
      res->end(s.data(), s.length());
    } else {
      // i guess this should be done more gracefully?
      res->end(nullptr, 0);
    }
  });

  h.onConnection([&h](uWS::WebSocket<uWS::SERVER> ws, uWS::HttpRequest req) {
    std::cout << "Connected!!!" << std::endl;
  });

  h.onDisconnection([&h](uWS::WebSocket<uWS::SERVER> ws, int code,
                         char *message, size_t length) {
    ws.close();
    std::cout << "Disconnected" << std::endl;
  });

  int port = 4567;
  if (h.listen(port)) {
    std::cout << "Listening to port " << port << std::endl;
  } else {
    std::cerr << "Failed to listen to port" << std::endl;
    return -1;
  }
  h.run();
}
