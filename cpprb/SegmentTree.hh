#ifndef YMD_SEGMENTTREE_HH
#define YMD_SEGMENTTREE_HH 1

#include <type_traits>
#include <functional>
#include <utility>
#include <vector>
#include <set>
#include <atomic>
#include <memory>

namespace ymd {
  inline constexpr auto PowerOf2(const std::size_t n) noexcept {
    auto m = std::size_t(1);
    while(m < n){ m *= 2; }
    return m;
  }

  template<typename T,bool MultiThread = false>
  class SegmentTree {
  private:
    using F = std::function<T(T,T)>;
    const std::size_t buffer_size;
    T* buffer;
    std::shared_ptr<T[]> view;
    F f;
    std::atomic_bool *any_changed;
    std::shared_ptr<std::atomic_bool> any_changed_view;

    auto _reduce(const std::size_t start,const std::size_t end,std::size_t index,
		 const std::size_t region_s,const std::size_t region_e) const {
      if((start <= region_s) && (region_e <= end)){
	return buffer[index];
      }

      auto region_m = (region_s + region_e)/2;

      if(end <= region_m){
	return _reduce(start,end,2*index+1,region_s,region_m);
      }

      if(region_m <= start){
	return _reduce(start,end,2*index+2,region_m,region_e);
      }

      return f(_reduce(start,end,2*index+1,region_s,region_m),
	       _reduce(start,end,2*index+2,region_m,region_e));
    }

    constexpr std::size_t parent(std::size_t node) const {
      return node ? (node - 1)/2: node;
    }

    constexpr auto child_left(std::size_t node) const {
      return 2 * node + 1;
    }

    constexpr auto child_right(std::size_t node) const {
      return 2 * node + 2;
    }

    auto access_index(std::size_t i) const {
      return buffer_size + i - 1;
    }

    bool update_buffer(std::size_t i){
      auto tmp = buffer[i];
      buffer[i] = f(buffer[child_left(i)],buffer[child_right(i)]);
      return tmp != buffer[i];
    }

    void update_init(){
      for(std::size_t i = access_index(0) -1, end = -1; i != end; --i){
	update_buffer(i);
      }
      if constexpr (MultiThread){
	any_changed->store(false,std::memory_order_release);
      }
    }

  public:
    SegmentTree(std::size_t n,F f, T v = T{0},
		T* buffer_ptr = nullptr,
		bool* any_changed_ptr = nullptr,
		bool initialize = true)
      : buffer_size(n),
	buffer(buffer_ptr),
	view{},
	f(f),
	any_changed{(std::atomic_bool*)any_changed_ptr},
	any_changed_view{}
    {
      if(!buffer){
	buffer = new T[2*n-1];
	view.reset(buffer);
      }

      if constexpr (MultiThread){
	if(!any_changed){
	  any_changed = new std::atomic_bool{true};
	  any_changed_view.reset(any_changed);
	}
      }

      if(initialize){
	std::fill_n(buffer+access_index(0),n,v);

	update_init();
      }
    }
    SegmentTree(): SegmentTree{2,[](auto a,auto b){ return a+b; }} {}
    SegmentTree(const SegmentTree&) = default;
    SegmentTree(SegmentTree&&) = default;
    SegmentTree& operator=(const SegmentTree&) = default;
    SegmentTree& operator=(SegmentTree&&) = default;
    ~SegmentTree() = default;

    T get(std::size_t i) const {
      return buffer[access_index(i)];
    }

    void set(std::size_t i,T v){
      auto n = access_index(i);
      buffer[n] = std::move(v);

      if constexpr (MultiThread){
	any_changed->store(true,std::memory_order_release);
      }else{
	constexpr const std::size_t zero = 0;
	auto updated = true;
	while((n != zero) && updated){
	  n = parent(n);
	  updated = update_buffer(n);
	}
      }
    }

    template<typename F,
	     typename std::enable_if<!(std::is_convertible_v<F,T>),
				     std::nullptr_t>::type = nullptr>
    void set(std::size_t i,F&& f,std::size_t N,std::size_t max = std::size_t(0)){
      constexpr const std::size_t zero = 0;
      if(zero == max){ max = buffer_size; }

      if constexpr (MultiThread){
	if(N){ any_changed->store(true,std::memory_order_release); }
      }

      while(N){
	auto copy_N = std::min(N,max-i);
	std::generate_n(buffer+access_index(i),copy_N,f);

	if constexpr (!MultiThread){
	  for(auto n = std::size_t(0); n < copy_N; ++n){
	    auto _i = access_index(i+n);
	    auto updated = true;
	      while((_i != zero) && updated){
		_i = parent(_i);
		updated = update_buffer(_i);
	    }
	  }
	}

	N = (N > copy_N) ? N - copy_N: zero;
	i = zero;
      }
    }

    void set(std::size_t i,T v,std::size_t N,std::size_t max = std::size_t(0)){
      set(i,[=](){ return v; },N,max);
    }

    auto reduce(std::size_t start,std::size_t end) {
      // Operation on [start,end)  # buffer[end] is not included
      if constexpr (MultiThread){
	if(any_changed->load(std::memory_order_acquire)){
	  update_all();
	}
      }
      return _reduce(start,end,0,0,buffer_size);
    }

    auto largest_region_index(std::function<bool(T)> condition,
			      std::size_t n=std::size_t(0),
			      T init = T{0}) {
      // max index of reduce( [0,index) ) -> true

      constexpr const std::size_t zero = 0;
      constexpr const std::size_t one  = 1;
      constexpr const std::size_t two  = 2;

      if constexpr (MultiThread){
	if(any_changed->load(std::memory_order_acquire)){
	  update_all();
	}
      }

      if(n == zero){ n = buffer_size; }
      auto b = zero;

      if(condition(buffer[b])){ return n-1; }

      auto min = zero;
      auto max = buffer_size;
      auto cond = condition;
      auto red = init;

      while(max - min > one){
	auto b_left = child_left(b);
	if(cond(buffer[b_left])){
	  min = (min + max) / two;
	  red = f(red, buffer[b_left]);
	  cond = [=](auto v){ return condition(f(red,v)); };
	  b = child_right(b);
	}else{
	  max = (min + max) / two;
	  b = b_left;
	}
      }

      return std::min(min, n-1);
    }

    void clear(T v = T{0}){
      std::fill(buffer + access_index(0), buffer + access_index(buffer_size), v);
      update_all();
    }
  };
}
#endif // YMD_SEGMENTTREE_HH
