#include <iostream>

#include <SegmentTree.hh>

int main(){

  auto st = ymd::SegmentTree<double>(16,[](auto a,auto b){ return a + b; });

  for(auto i = 0ul; i < 16ul; ++i){
    st.set(i,i*1.0);
    std::cout << i * 1.0 << " ";
  }
  std::cout << std::endl;

  std::cout << "[0,11): " << st.reduce(0,11) << std::endl;
  std::cout << "[13,15): " << st.reduce(13,15) << std::endl;

  return 0;
}
