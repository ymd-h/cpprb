#include <algorithm>
#include <iostream>
#include <iterator>
#include <chrono>
#include <vector>

#include <SegmentTree.hh>
#include <ReplayBuffer.hh>

using PER = ymd::CppPrioritizedSampler<float>;
using MPPER = ymd::CppThreadSafePrioritizedSampler<float>;

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
  bench([&,i=100]() mutable { sum.set(100*(i++),
				      [j=0]()mutable{ return j++; },
				      100,
				      buffer_size); }, 100, "sum1.set B: ");
  bench([&,i=20]() mutable { sum.set(1000*(i++),
				     [j=0]()mutable{ return j++; },
				     1000,
				      buffer_size); }, 10, "sum1.set C: ");
  bench([&]() mutable { sum.reduce(0, 30000); }, 1, "sum1.red A: ");
  bench([&]() mutable { sum.reduce(0, 30000); }, 1, "sum1.red B: ");
  bench([&,i=0]() mutable {
    sum.largest_region_index([&](auto v){ return v <= 79.8*(i++); }, 30000);
  },
    10000, "sum1.lridx: ");

  bench([&,i=0, j=0]() mutable { sum2.set(i++, j++); }, 10000, "sum2.set A: ");
  bench([&,i=100]() mutable { sum2.set(100*(i++),
				       [j=0]()mutable{ return j++; },
				       100,
				       buffer_size); }, 100, "sum2.set B: ");
  bench([&,i=20]() mutable { sum2.set(1000*(i++),
				      [j=0]()mutable{ return j++; },
				      1000,
				      buffer_size); }, 10, "sum2.set C: ");
  bench([&]() mutable { sum2.reduce(0, 30000); }, 1, "sum2.red A: ");
  bench([&]() mutable { sum2.reduce(0, 30000); }, 1, "sum2.red B: ");
  bench([&,i=0]() mutable {
    sum2.largest_region_index([&](auto v){ return v <= 79.8*(i++); }, 30000);
  },
    10000, "sum2.lridx: ");

  std::cout << sum.get(1) << " " << sum2.get(1) << std::endl;
  std::cout << sum.get(10001) << " " << sum2.get(10001) << std::endl;

  //

  constexpr const auto alpha = 0.5, beta = 0.4;
  auto per = PER(buffer_size, alpha);
  auto mpper = MPPER(buffer_size, alpha);

  auto p = std::vector<float>{};
  p.reserve(10000);
  std::generate_n(std::back_inserter(p), 10000,
		  [i=0]() mutable { return 0.02*(i++ % 321); });

  auto indexes = std::vector<size_t>{};
  indexes.reserve(32);
  auto weights = std::vector<float>{};
  weights.reserve(32);

  bench([&, i=0,j=0]() mutable { per.set_priorities(i++, 0.02*(j++ % 321)); },
	10000, "  PER.add1: ");
  bench([&, i=100,j=0]() mutable {
    per.set_priorities(100*(i++), p.data()+100*(j++), 100, buffer_size);
  },
    100, "  PER.addN: ");
  bench([&]() mutable { per.sample(32,beta,weights,indexes,20000); },
	1, "  PER.smpA: ");
  bench([&]() mutable { per.sample(32,beta,weights,indexes,20000); },
	1, "  PER.smpB: ");

  bench([&, i=0,j=0]() mutable { mpper.set_priorities(i++, 0.02*(j++ % 321)); },
	10000, "MPPER.add1: ");
  bench([&, i=100,j=0]() mutable {
    mpper.set_priorities(100*(i++),  p.data()+100*(j++), 100, buffer_size);
  },
    100, "MPPER.addN: ");
  bench([&]() mutable { mpper.sample(32,beta,weights,indexes,20000); },
	1, "MPPER.smpA: ");
  bench([&]() mutable { mpper.sample(32,beta,weights,indexes,20000); },
	1, "MPPER.smpB: ");

  return 0;
}
