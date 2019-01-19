#ifndef YMD_SEGMENTTREE_HH
#define YMD_SEGMENTTREE_HH 1

#include <type_traits>
#include <functional>
#include <utility>
#include <vector>

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
		 const std::size_t region_s,const std::size_t region_e){
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
  public:
    SegmentTree(std::size_t n,F f): size(n), buffer(2*n-1), f(f) {
      for(auto i = n-2, stop = 0ul - 1ul; i != stop ; --i){
	buffer[i] = f(buffer[child_left(i)],
		      buffer[child_right(i)]);
      }
    }
    SegmentTree(): SegmentTree{2,[](auto a,auto b){ return a+b; }} {}
    SegmentTree(const SegmentTree&) = default;
    SegmentTree(SegmentTree&&) = default;
    SegmentTree& operator=(const SegmentTree&) = default;
    SegmentTree& operator=(SegmentTree&&) = default;
    ~SegmentTree() = default;

    auto get(std::size_t i){
      return buffer[access_index(i)];
    }

    void set(std::size_t i,T v){
      auto n = access_index(i);
      buffer[n] = std::move(v);

      do {
	n = parent(n);

	buffer[n] = f(buffer[child_left(n)],
		      buffer[child_right(n)]);

      } while(n != 0ul);
    }

    auto reduce(std::size_t start,std::size_t end){
      // Operation on [start,end)  # buffer[end] is not included

      return _reduce(start,end,0,0,size);
    }

    auto largest_region_index(std::vector<bool(T)> condition){
      // max index of reduce( [0,index) ) -> true

      auto min = 0ul;
      auto max = buffer.size();

      auto index = (min + max)/2ul;

      while(max - min > 1ul){
	( condition(reduce(0ul,index)) ? min : max ) = index;
	index = (min + max)/2ul;
      }

      return index;
    }
  };
}
#endif // YMD_SEGMENTTREE_HH
