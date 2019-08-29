#include <uWS/uWS.h>
#include <iostream>
#include "json.hpp"
#include "PID.h"
#include <math.h>

// for convenience
using json = nlohmann::json;

// For converting back and forth between radians and degrees.
constexpr double pi() { return M_PI; }
double deg2rad(double x) { return x * pi() / 180; }
double rad2deg(double x) { return x * 180 / pi(); }

// Checks if the SocketIO event has JSON data.
// If there is data the JSON object in string format will be returned,
// else the empty string "" will be returned.
std::string hasData(std::string s) {
  auto found_null = s.find("null");
  auto b1 = s.find_first_of("[");
  auto b2 = s.find_last_of("]");
  if (found_null != std::string::npos) {
    return "";
  }
  else if (b1 != std::string::npos && b2 != std::string::npos) {
    return s.substr(b1, b2 - b1 + 1);
  }
  return "";
}

int main()
{
  uWS::Hub h;

  // comment speed_pid if only using steer_pid
  PID steer_pid;
  PID speed_pid;
  // Initialize the pid variable.
  // TODO: Tweak initial Kp, Ki, Kd values.
  
  double steer_Kp = 0.13; //0.1 for only steer_pid without speed_pid
  double steer_Ki = 0.0001; //0.005 for only steer_pid without speed_pid
  double steer_Kd = 1.0; //4.0 for only steer_pid without speed_pid
  steer_pid.Init(steer_Kp, steer_Ki, steer_Kd);
  
  double speed_Kp = 0.1;
  double speed_Ki = 0.002;
  double speed_Kd = 0.0;
  speed_pid.Init(speed_Kp, speed_Ki, speed_Kd);

  h.onMessage([&pid](uWS::WebSocket<uWS::SERVER> ws, char *data, size_t length, uWS::OpCode opCode) {
    // "42" at the start of the message means there's a websocket message event.
    // The 4 signifies a websocket message
    // The 2 signifies a websocket event
    if (length && length > 2 && data[0] == '4' && data[1] == '2')
    {
      auto s = hasData(std::string(data));
      if (s != "") {
        auto j = json::parse(s);
        std::string event = j[0].get<std::string>();
        if (event == "telemetry") {
          // j[1] is the data JSON object
          double cte = std::stod(j[1]["cte"].get<std::string>());
          double speed = std::stod(j[1]["speed"].get<std::string>());
          double angle = std::stod(j[1]["steering_angle"].get<std::string>());
          double steer_value;
          /*
          * TODO: Calcuate steering value here, remember the steering value is
          * [-1, 1].
          * NOTE: Feel free to play around with the throttle and speed. Maybe use
          * another PID controller to control the speed!
          */
          /* Was going to add twiddle but it worked without twiddle
          if (num_measurements % steps_between_twiddles == 0) {
            pid.Twiddle();
          }
           */
          steer_pid.UpdateError(cte);
          steer_value = steer_pid.TotalError();
          std::cout << "Steer: " << steer_value << std::endl;
          if (steer_value > 1) 
          {
            steer_value = 1;
          }
          else if (steer_value < -1) 
          {
            steer_value = -1;
          }
          
          double throttle_value;
          double desired_speed = 30;
          double error_speed = speed - desired_speed;
          speed_pid.UpdateError(error_speed);
          throttle_value = speed_pid.TotalError();
          std::cout << "Throttle: " << throttle_value << std::endl;
          if (throttle_value > 1) 
          {
            throttle_value = 1;
          }
          else if (throttle_value < -1) 
          {
            throttle_value = -1;
          }
          // DEBUG
          std::cout << "CTE: " << cte << " Steering Value: " << steer_value << std::endl;

          json msgJson;
          msgJson["steering_angle"] = steer_value;
          msgJson["throttle"] = 0.3;
          auto msg = "42[\"steer\"," + msgJson.dump() + "]";
          std::cout << msg << std::endl;
          ws.send(msg.data(), msg.length(), uWS::OpCode::TEXT);
        }
      } else {
        // Manual driving
        std::string msg = "42[\"manual\",{}]";
        ws.send(msg.data(), msg.length(), uWS::OpCode::TEXT);
      }
    }
  });

  // We don't need this since we're not using HTTP but if it's removed the program
  // doesn't compile :-(
  h.onHttpRequest([](uWS::HttpResponse *res, uWS::HttpRequest req, char *data, size_t, size_t) {
    const std::string s = "<h1>Hello world!</h1>";
    if (req.getUrl().valueLength == 1)
    {
      res->end(s.data(), s.length());
    }
    else
    {
      // i guess this should be done more gracefully?
      res->end(nullptr, 0);
    }
  });

  h.onConnection([&h](uWS::WebSocket<uWS::SERVER> ws, uWS::HttpRequest req) {
    std::cout << "Connected!!!" << std::endl;
  });

  h.onDisconnection([&h](uWS::WebSocket<uWS::SERVER> ws, int code, char *message, size_t length) {
    ws.close();
    std::cout << "Disconnected" << std::endl;
  });

  int port = 4567;
  if (h.listen(port))
  {
    std::cout << "Listening to port " << port << std::endl;
  }
  else
  {
    std::cerr << "Failed to listen to port" << std::endl;
    return -1;
  }
  h.run();
}
