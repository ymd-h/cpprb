#include <iostream>
#include <cassert>
#include <type_traits>

#include <SegmentTree.hh>

template<typename T1,typename T2>
auto Equal(T1&& v,T2&& expected){
  if(v != expected){
    std::cout << std::endl
	      << "Assert Equal: " << v << " != " << expected << std::endl;
    assert(v == expected);
  }

  return v;
}

template<typename T1,typename T2>
auto AlmostEqual(T1&& v,T2&& expected, std::common_type_t<T1,T2>&& eps = 1e-5){
  if(std::abs(v - expected) > eps){
    std::cout << std::endl
	      << "Assert AlmostEqual: " << v << " != " << expected << std::endl;
    assert(std::abs(v - expected) <= eps);
  }

  return v;
}

int main(){
  constexpr auto buffer_size = 16;

  auto st = ymd::SegmentTree<double>(buffer_size,[](auto a,auto b){ return a + b; });

  for(auto i = 0ul; i < 16ul; ++i){
    st.set(i,i*1.0);
    std::cout << i * 1.0 << " ";
  }
  std::cout << std::endl;

  std::cout << "[0,11): " << AlmostEqual(st.reduce(0,11),55) << std::endl;
  std::cout << "[13,15): " << AlmostEqual(st.reduce(13,15),27) << std::endl;

  std::cout << "[0,x) <= 7: x = "
	    << Equal(st.largest_region_index([](auto v){ return v <=7; }),4)
	    << std::endl;

  std::cout << std::endl;

  constexpr auto set_index = 12;
  constexpr auto set_value = 5;
  constexpr auto set_size = 10;

  st.set(set_index,set_value,set_size);
  for(auto i = 0ul; i < 16ul; ++i){
    std::cout << AlmostEqual(st.get(i),(i < set_size - (buffer_size - set_index) ||
					set_index <= i) ? set_value : i)
	      << " ";
  }
  std::cout << std::endl;

  std::cout << "[0,11): "
	    << AlmostEqual(st.reduce(0,11),70) << std::endl;
  std::cout << "[13,15): " << AlmostEqual(st.reduce(13,15),10) << std::endl;

  std::cout << "[0,x) <= 7: x = "
	    << Equal(st.largest_region_index([](auto v){ return v <=7; }),1)
	    << std::endl;

  return 0;
}
