#ifndef YMD_REPLAY_BUFFER_HH
#define YMD_REPLAY_BUFFER_HH 1

#include <cmath>
#include <vector>
#include <random>
#include <utility>
#include <deque>
#include <tuple>
#include <functional>
#include <type_traits>

#include "SegmentTree.hh"

template<typename T> struct UnderlyingType {
  using type = T;
  static constexpr auto size(T&){ return 1ul; }
};
template<typename T> struct UnderlyingType<std::vector<T>>{
  using type = T;
  static auto size(std::vector<T>& v){ return v.size(); }
};

namespace ymd {
  template<typename Observation,typename Action,typename Reward,typename Done>
  class ReplayBuffer {
  public:
    using buffer_t = std::deque<std::tuple<Observation,Action,Reward,Observation,Done>>;
    using rand_t = std::uniform_int_distribution<std::size_t>;
  private:
    const std::size_t size;
    buffer_t buffer;
    std::mt19937 g;

    template<typename T>
    void flatten_push_back(T&& v,
			   std::vector<std::remove_reference_t<T>>& to){
      to.push_back(std::forward<T>(v));
    }

    template<typename T>
    void flatten_push_back(const std::vector<T>& v,std::vector<T>& to){
      std::copy(v.begin(),v.end(),std::back_inserter(to));
    }

    template<typename T>
    void flatten_push_back(std::vector<T>&& v,std::vector<T>& to){
      std::move(v.begin(),v.end(),std::back_inserter(to));
    }

  public:
    ReplayBuffer(std::size_t n): size(n),g{std::random_device{}()} {}
    ReplayBuffer(): ReplayBuffer{1} {}
    ReplayBuffer(const ReplayBuffer&) = default;
    ReplayBuffer(ReplayBuffer&&) = default;
    ReplayBuffer& operator=(const ReplayBuffer&) = default;
    ReplayBuffer& operator=(ReplayBuffer&&) = default;
    ~ReplayBuffer() = default;

    void add(Observation obs,Action act,Reward rew,Observation next_obs,Done done){
      if(size == buffer.size()){
	buffer.pop_front();
      }
      buffer.emplace_back(std::move(obs),std::move(act),std::move(rew),std::move(next_obs),std::move(done));
    }

    void sample(std::size_t batch_size,
		std::vector<typename UnderlyingType<Observation>::type>& obs,
		std::vector<typename UnderlyingType<Action>::type>& act,
		std::vector<typename UnderlyingType<Reward>::type>& rew,
		std::vector<typename UnderlyingType<Observation>::type>& next_obs,
		std::vector<typename UnderlyingType<Done>::type>& done,
		...){
      obs.resize(0);
      act.resize(0);
      rew.resize(0);
      next_obs.resize(0);
      done.resize(0);

      obs.reserve(batch_size *
		  UnderlyingType<Observation>::size(std::get<0>(buffer[0])));
      act.reserve(batch_size *
		  UnderlyingType<Action>::size(std::get<1>(buffer[0])));
      rew.reserve(batch_size *
		  UnderlyingType<Reward>::size(std::get<2>(buffer[0])));
      next_obs.reserve(batch_size *
		       UnderlyingType<Observation>::size(std::get<3>(buffer[0])));
      done.reserve(batch_size *
		   UnderlyingType<Done>::size(std::get<4>(buffer[0])));

      auto random = [this,d=rand_t{0,buffer.size()-1}]()mutable{ return d(this->g); };

      for(auto i = 0ul; i < batch_size; ++i){
	// Done can be bool, so that "std::tie(...,d[i]) = buffer[random()]" may fail.
	auto [o,a,r,no,d] = buffer[random()];

	flatten_push_back(std::move(o),obs);
	flatten_push_back(std::move(a),act);
	flatten_push_back(std::move(r),rew);
	flatten_push_back(std::move(no),next_obs);
	flatten_push_back(std::move(d),done);
      }

    }

    void sample(std::size_t batch_size,
		std::vector<Observation>& obs,
		std::vector<Action>& act,
		std::vector<Reward>& rew,
		std::vector<Observation>& next_obs,
		std::vector<Done>& done){
      obs.resize(0);
      act.resize(0);
      rew.resize(0);
      next_obs.resize(0);
      done.resize(0);

      obs.reserve(batch_size);
      act.reserve(batch_size);
      rew.reserve(batch_size);
      next_obs.reserve(batch_size);
      done.reserve(batch_size);

      auto random = [this,d=rand_t{0,buffer.size()-1}]()mutable{ return d(this->g); };

      for(auto i = 0ul; i < batch_size; ++i){
	// Done can be bool, so that "std::tie(...,d[i]) = buffer[random()]" may fail.
	auto [o,a,r,no,d] = buffer[random()];

	obs.push_back(std::move(o));
	act.push_back(std::move(a));
	rew.push_back(std::move(r));
	next_obs.push_back(std::move(no));
	done.push_back(std::move(d));
      }
    }

    auto sample(std::size_t batch_size){
      std::vector<Observation> obs{},next_obs{};
      std::vector<Action> act{};
      std::vector<Reward> rew{};
      std::vector<Done> done{};

      sample(batch_size,obs,act,rew,next_obs,done);

      return std::make_tuple(obs,act,rew,next_obs,done);
    }
  };

  template<typename Observation,typename Action,typename Reward,typename Done,
	   typename Priority>
  class PrioritizedReplayBuffer
    :public ReplayBuffer<Observation,Action,Reward,Done> {
  private:
    Priority alpha;
    Priority max_priority;
    SegmentTree<Priority> sum;
    SegmentTree<Priority> min;
    std::size_t next_idx
    auto sample_proportional(std::size_t batch_size) const {
      auto res = std::vector<std::size_t>{};
      res.reserve(batch_size);

      auto every_range_len = Priority{1.0} * sum.reduce(0,buffer.size()) / batch_size;

      std::generate_n(std::back_inserter(res),batch_size,
		      [=,i=0ul
		       d=std::uniform_real_distribution<Priority>{}]()mutable{
			auto mass = (d(this->g) + (i++))*every_range_len;
			return this->sum.largest_region_index([=](auto v){
								return v <= mass;
							      });
		      });
      return res;
    }
  public:
    PrioritizedReplayBuffer(std::size_t n,Priority alpha)
      : ReplayBuffer{n},
	alpha{std::max(alpha,Priority{0.0})},
	max_priority{1.0},
	sum{PowerOf2(n),[](auto a,auto b){ return a+b; }},
	min{PowerOf2(n),[zero = Priority{0}](Priority a,Priority b){
			  return ((zero == a) ? b:
				  (zero == b) ? a:
				  std::min(a,b));
			}},
	next_idx{0} {}
    PrioritizedReplayBuffer() : PrioritizedReplayBuffer{1,0.0} {}
    PrioritizedReplayBuffer(const PrioritizedReplayBuffer&) = default;
    PrioritizedReplayBuffer(PrioritizedReplayBuffer&&) = default;
    PrioritizedReplayBuffer& operator=(const PrioritizedReplayBuffer&) = default;
    PrioritizedReplayBuffer& operator=(PrioritizedReplayBuffer&&) = default;
    ~PrioritizedReplayBuffer() = default;

    void add(Observation obs,Action act,Reward rew,
	     Observation next_obs,Done done) override {
      this->ReplayBuffer::add(std::move(obs),std::move(act),std::move(rew),
			      std::move(next_obs),std::move(done));

      auto v = std::pow(max_priority,alpha);
      sum.set(next_idx,v);
      min.set(next_idx,v);

      if(size == ++next_idx){ next_idx = 0ul; }
    }

    auto sample(std::size_t batch_size,Priority beta){
      beta = std::max(beta,Priority{0});

      auto indexes = sample_proportional(batch_size);

      auto b_size = buffer.size();
      auto inv_sum = Priority{1.0} / sum.reduce(0,b_size);
      auto p_min = min.reduce(0,b_size) * inv_sum;
      auto inv_max_weight = Priority{1.0} / std::pow(p_min * b_size(),-beta);

      auto weights = std::vector<Priority>{};
      weights.reserve(batch_size);

      std::transform(indexes.begin(),indexes.end(),std::back_inserter(weights),
		     [=](auto idx){
		       auto p_sample = this->sum.get(idx) * inv_sum;
		       return std::pow(p_sample*b_size,-beta)*inv_max_weight;
		     });

      auto samples = this->ReplayBuffer::sample(batch_size);
      return std::tuple_cat(samples,std::make_tuple(weights,indexes));
    }
  };
}
#endif // YMD_REPLAY_BUFFER_HH
