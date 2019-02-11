#ifndef YMD_UNITTEST_HH
#define YMD_UNITTEST_HH 1

namespace ymd {
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
