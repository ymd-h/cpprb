#ifndef YMD_UNITTEST_HH
#define YMD_UNITTEST_HH 1

#include <string>

#define EQUAL(a,b) ymd::Equal((a),(b),std::to_string(#a),std::to_string(#b))
#define ALMOST_EQUAL(a,b) ymd::AlmostEqual((a),(b),1e-5,std::to_string(#a),std::to_string(#b))

namespace ymd {
  template<typename F>
  inline auto timer(F&& f,std::size_t N){
    auto start = std::chrono::high_resolution_clock::now();

    for(std::size_t i = 0ul; i < N; ++i){ f(); }

    auto end = std::chrono::high_resolution_clock::now();
    auto elapsed = end - start;

    auto s = std::chrono::duration_cast<std::chrono::seconds>(elapsed);
    auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(elapsed);
    auto us = std::chrono::duration_cast<std::chrono::microseconds>(elapsed);
    auto ns = std::chrono::duration_cast<std::chrono::nanoseconds>(elapsed);
    std::cout << s.count() << "s "
	      << ms.count() - s.count() * 1000 << "ms "
	      << us.count() - ms.count() * 1000 << "us "
	      << ns.count() - us.count() * 1000 << "ns"
	      << std::endl;
  }

  template<typename T1,typename T2>
  auto Equal(T1&& v,T2&& expected,std::string lhs="",std::string rhs=""){
    if(v != expected){
      std::cout << std::endl
		<< "Fail Equal: "
		<< lhs.size() ? lhs + " -> ": ""
		<< v
		<< " != "
		<< rhs.size() ? rhs + " -> ": ""
		<< expected
		<< std::endl;
      assert(v == expected);
    }
    return v;
  }

  template<typename T1,typename T2>
  auto AlmostEqual(T1&& v,T2&& expected, std::common_type_t<T1,T2>&& eps = 1e-5,
		   std::string lhs="",std::string rhs=""){
    if(std::abs(v - expected) > eps){
      std::cout << std::endl
		<< "Assert AlmostEqual: "
		<< lhs.size() ? lhs + " -> ": ""
		<< v
		<< " != "
		<< rhs.size() ? rhs + " -> ": ""
		<< expected
		<< std::endl;
      assert(std::abs(v - expected) <= eps);
    }
    return v;
  }

  template<typename T>
  void show_vector(T v,std::string name){
    std::cout << name << ": ";
    for(auto ve: v){ std::cout << ve << " "; }
    std::cout << std::endl;
  }

  template<typename T>
  void show_vector_of_vector(T v,std::string name){
    std::cout << name << ": " << std::endl;
    for(auto ve: v){
      std::cout << " ";
      for(auto vee: ve){ std::cout << vee << " "; }
      std::cout << std::endl;
    }
    std::cout << std::endl;
  }

  template<typename T>
  void show_pointer(T ptr,std::size_t N,std::string name){
    auto v = std::vector<std::remove_pointer_t<T>>(ptr,ptr+N);
    show_vector(v,name);
  }
}
#endif // YMD_UNITTEST_HH
