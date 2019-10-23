#pragma once
#include <memory>
#include <memory>
#include <map>
#ifdef __linux__
#include "linux-perf-events.h"
#else
#define PERF_TYPE_HARDWARE 0
#endif

struct TypeAndConfig {
  uint64_t type, config;
  TypeAndConfig(uint64_t type, uint64_t config) : type{type}, config{config} {}
  bool operator<(const TypeAndConfig& rhs) const {
    return type < rhs.type || (type == rhs.type && config < rhs.config);
  }
};

std::string to_string(const TypeAndConfig &e) {
  return "[type=" + std::to_string(e.type) + ", config=" + std::to_string(e.config) + "]";
}

class LinuxEventsWrapper {
  public:
    LinuxEventsWrapper(const std::vector<TypeAndConfig> events) {
#ifdef __linux__
      for (auto& e: events) {
        event_obj.emplace(e, std::make_shared<LinuxEvents>(e.type, e.config));
        event_res.emplace(e, 0);
      }
#endif
    }
    void start() {
#ifdef __linux__
      for (const auto& [_, ptr]: event_obj) {
        ptr->start();
      }
#endif
    }
    void end() {
#ifdef __linux__
      for (const auto& [ecode, ptr]: event_obj) {
        event_res[ecode] = ptr->end();
      }
#endif
    }
    // Throws an exception if the code is not present
    unsigned long get_result(TypeAndConfig e) {
#ifdef __linux__
      if (event_res.count(e) == 0) {
        printf("Not found: %s\n", to_string(e).c_str());
      }
      return event_res.at(e);
#else
      return 0;
#endif
    }
  private:
#ifdef __linux__
    std::map<TypeAndConfig, std::shared_ptr<LinuxEvents>> event_obj;
    std::map<TypeAndConfig, unsigned long> event_res;
#endif
};
