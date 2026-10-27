#include <csignal>
#include <iostream>

#include "node.hpp"

int main(int argc, char** argv) {
  std::signal(SIGPIPE, SIG_IGN);
  try {
    pouw::Node node(pouw::parse_args(argc, argv));
    node.run();
  } catch (const std::exception& e) {
    std::cerr << "node: " << e.what() << '\n'
              << "usage: node --id N --port P --task task.json [--peers host:port,...]\n"
                 "            [--store DIR] [--run-dir DIR] [--python PATH] [--worker PATH]\n"
                 "            [--device cpu|mps] [--q 2] [--alpha 0.05] [--verify-a]\n"
                 "            [--lazy] [--lazy-delay SEC] [--max-height H] [--api-port P]\n";
    return 1;
  }
  return 0;
}
