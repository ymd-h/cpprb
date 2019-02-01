#ifndef YMD_SEGMENTTREE_HH
#define YMD_SEGMENTTREE_HH 1

#include <type_traits>
#include <functional>
#include <utility>
#include <vector>
#include <set>

namespace ymd {
  inline constexpr auto PowerOf2(std::size_t n) noexcept {
    auto m = 1ul;
    while(m < n){ m *= 2; }
    return m;
  }

  template<typename T>
  class SegmentTree {
  private:
    using F = std::function<T(T,T)>;
    const std::size_t size;
    std::vector<T> buffer;
    F f;

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

    auto parent(std::size_t node) const {
      return (node - 1)/2ul;
    }

    auto child_left(std::size_t node) const {
      return 2 * node + 1;
    }

    auto child_right(std::size_t node) const {
      return 2 * node + 2;
    }

    auto access_index(std::size_t i) const {
      return size + i - 1;
    }

    void update_buffer(std::size_t i){
      buffer[i] = f(buffer[child_left(i)],buffer[child_right(i)]);
    }

    void update_all(){
      for(auto i = access_index(0) -1, end = 0ul - 1ul; i != end; --i){
	update_buffer(i);
      }
    }
  public:
    SegmentTree(std::size_t n,F f, T v = T{0}): size(n), buffer(2*n-1,v), f(f) {
      update_all();
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

      do {
	n = parent(n);
	update_buffer(n);
      } while(n != 0ul);
    }

    template<typename F,
	     typename std::enable_if<!(std::is_convertible_v<F,T>),
				     std::nullptr_t>::type = nullptr>
    void set(std::size_t i,F&& f,std::size_t N,std::size_t max = 0ul){
      if(0ul == max){ max = size; }

      std::set<std::size_t> will_update{};

      while(N){
	auto copy_N = std::min(N,max-i);
	std::generate_n(buffer.data()+access_index(i),copy_N,f);

	for(auto n = 0ul; n < copy_N; ++n){
	  will_update.insert(parent(access_index(i+n)));
	}

	N = (N > copy_N) ? N - copy_N: 0ul;
	i = 0ul;
      }


      while(!will_update.empty()){
	i = *(will_update.rbegin());
	update_buffer(i);
	will_update.erase(i);
	if(i){ will_update.insert(parent(i)); }
      }
    }

    void set(std::size_t i,T v,std::size_t N,std::size_t max = 0ul){
      set(i,[=](){ return v; },N,max);
    }

    auto reduce(std::size_t start,std::size_t end) const {
      // Operation on [start,end)  # buffer[end] is not included

      return _reduce(start,end,0,0,size);
    }

    auto largest_region_index(std::function<bool(T)> condition,
			      std::size_t n=0ul) const {
      // max index of reduce( [0,index) ) -> true

      auto min = 0ul;
      auto max = (0ul != n) ? n: size;

      auto index = (min + max)/2ul;

      while(max - min > 1ul){
	( condition(reduce(0ul,index)) ? min : max ) = index;
	index = (min + max)/2ul;
      }

      return index;
    }

    void clear(T v = T{0}){
      std::fill(buffer.begin() + access_index(0), buffer.end(), v);
      update_all();
    }
  };
}
#endif // YMD_SEGMENTTREE_HH
