#include <iostream>
#include <chrono>

#include <SegmentTree.hh>

auto bench = [](auto&& F, auto n, auto fmt=""){
  auto t1 = std::chrono::high_resolution_clock::now();
  for(auto i=0ul; i < n; ++i){ F(); }
  auto t2 = std::chrono::high_resolution_clock::now();

  std::cout << fmt
	    << std::chrono::duration_cast<std::chrono::nanoseconds>(t2-t1).count()
	    << std::endl;
 };


int main(int argc, char** argv){
  constexpr const auto buffer_size = 1000000ul;
  constexpr const auto size = ymd::PowerOf2(buffer_size);

  auto sum  = ymd::SegmentTree<float>(size, [](auto a, auto b){ return a+b; });
  auto sum2 = ymd::SegmentTree<float,true>(size, [](auto a, auto b){ return a+b; });

  bench([&,i=0, j=0]() mutable { sum.set(i++, j++); }, 10000, "sum1.set A: ");
  bench([&,i=0]() mutable { sum.set(100*(i++),
				    [j=0]()mutable{ return j++; },
				    100,
				    buffer_size); }, 100, "sum1.set B: ");
  bench([&]() mutable { sum.reduce(0, 20000); }, 1, "sum1.red A: ");
  bench([&]() mutable { sum.reduce(0, 20000); }, 1, "sum1.red B: ");
  bench([&,i=0]() mutable {
    sum.largest_region_index([&](auto v){ return v <= 79.8*(i++); }, 20000);
  },
    10000, "sum1.lridx: ");

  bench([&,i=0, j=0]() mutable { sum2.set(i++, j++); }, 10000, "sum2.set A: ");
  bench([&,i=0]() mutable { sum2.set(100*(i++),
				     [j=0]()mutable{ return j++; },
				     100,
				     buffer_size); }, 100, "sum2.set B: ");
  bench([&]() mutable { sum2.reduce(0, 20000); }, 1, "sum2.red A: ");
  bench([&]() mutable { sum2.reduce(0, 20000); }, 1, "sum2.red B: ");
  bench([&,i=0]() mutable {
    sum2.largest_region_index([&](auto v){ return v <= 79.8*(i++); }, 20000);
  },
    10000, "sum2.lridx: ");

  std::cout << sum.get(1) << " " << sum2.get(1) << std::endl;
  std::cout << sum.get(1001) << " " << sum2.get(1001) << std::endl;

  return 0;
}
