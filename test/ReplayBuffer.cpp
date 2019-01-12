#include <iostream>

#include <ReplayBuffer.hh>

int main(){
  auto rb = ymd::ReplayBuffer<std::vector<double>,
			      std::vector<double>,
			      double,bool>{15ul};

  return 0;
}
