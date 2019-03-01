#include <iostream>
#include <cassert>
#include <type_traits>
#include <future>
#include <thread>

#include <SegmentTree.hh>

#include "unittest.hh"

void multi_thread_test(){
  constexpr auto buffer_size = 16ul;
  auto st = ymd::SegmentTree<double,true>(buffer_size,
					  [](auto a,auto b){ return a+b; });
  const auto cores = std::thread::hardware_concurrency();

  auto futures = std::vector<std::future<void>>{};

  std::generate_n(std::back_inserter(futures),cores,
		  [&,index = 0,v = 2] () mutable {
		    index = (index + 3) % buffer_size;
		    v *= 2;
		    return std::async(std::launch::async,
				      [&st](auto index,auto v){
					st.set(index,v,5);
				      },index,v);
		  });

  for(auto& f : futures){ f.wait(); }
  for(auto i = 0ul; i < buffer_size; ++i){
    std::cout << st.get(i) << " ";
  }
  std::cout << std::endl;
}

int main(){
  constexpr auto buffer_size = 16;

  auto st = ymd::SegmentTree<double>(buffer_size,[](auto a,auto b){ return a + b; });

  for(auto i = 0ul; i < 16ul; ++i){
    st.set(i,i*1.0);
    std::cout << i * 1.0 << " ";
  }
  std::cout << std::endl;

  std::cout << "[0,11): " << ymd::AlmostEqual(st.reduce(0,11),55) << std::endl;
  std::cout << "[13,15): " << ymd::AlmostEqual(st.reduce(13,15),27) << std::endl;

  std::cout << "[0,x) <= 7: x = "
	    << ymd::Equal(st.largest_region_index([](auto v){ return v <=7; }),4)
	    << std::endl;

  std::cout << std::endl;

  constexpr auto set_index = 12;
  constexpr auto set_value = 5;
  constexpr auto set_size = 10;

  st.set(set_index,set_value,set_size);
  for(auto i = 0ul; i < 16ul; ++i){
    std::cout << ymd::AlmostEqual(st.get(i),
				  (i < set_size - (buffer_size - set_index) ||
				   set_index <= i) ? set_value : i)
	      << " ";
  }
  std::cout << std::endl;

  std::cout << "[0,11): "
	    << ymd::AlmostEqual(st.reduce(0,11),70) << std::endl;
  std::cout << "[13,15): " << ymd::AlmostEqual(st.reduce(13,15),10) << std::endl;

  std::cout << "[0,x) <= 7: x = "
	    << ymd::Equal(st.largest_region_index([](auto v){ return v <=7; }),1)
	    << std::endl;

  multi_thread_test();

  return 0;
}
