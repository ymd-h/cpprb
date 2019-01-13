#ifndef YMD_SEGMENTTREE_HH
#define YMD_SEGMENTTREE_HH 1

#include <type_traits>
#include <functional>
#include <utility>
#include <vector>

namespace ymd {

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
  public:
    SegmentTree(std::size_t n,F f): size(n), buffer(2*n-1), f(f) {
      for(auto i = n-2, stop = 0ul - 1ul; i != stop ; --i){
	buffer[i] = f(2*i+1,2*i+2);
      }
    }
    SegmentTree(): SegmentTree{2} {}
    SegmentTree(const SegmentTree&) = default;
    SegmentTree(SegmentTree&&) = default;
    SegmentTree& operator=(const SegmentTree&) = default;
    SegmentTree& operator=(SegmentTree&&) = default;
    ~SegmentTree() = default;

    auto get(std::size_t i){
      return buffer[size+i-1];
    }

    void set(std::size_t i,T v){
      auto n = size+i-1;
      buffer[n] = std::move(v);

      do {
	n = (n-1)/2;

	buffer[n] = f(buffer[2*n+1],buffer[2*n+2]);

      } while(n != 0ul);
    }

    auto reduce(std::size_t start,std::size_t end){
      // Operation on [start,end)  # buffer[end] is not included

      return _reduce(start,end,0,0,size);
    }
  };
}
#endif // YMD_SEGMENTTREE_HH
