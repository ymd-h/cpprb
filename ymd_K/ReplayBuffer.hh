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

namespace ymd {
  template<typename Observation,typename Action,typename Reward,typename Done>
  class ReplayBuffer {
  public:
    using rand_t = std::uniform_int_distribution<std::size_t>;
  private:
    const std::size_t capacity;
    std::size_t size;
    std::size_t obs_dim;
    std::size_t act_dim;
    std::size_t next_index;
    std::vector<Observation> obs_buffer;
    std::vector<Action> act_buffer;
    std::vector<Reward> rew_buffer;
    std::vector<Observation> next_obs_buffer;
    std::vector<Done> done_buffer;

    void store_buffer(Observation* obs,
		      Action* act,
		      Reward* rew,
		      Observation* next_obs,
		      Done* done,
		      std::size_t shift,
		      std::size_t N){
      obs += shift * obs_dim;
      act += shift * act_dim;
      rew += shift;
      next_obs += shift * obs_dim;
      done += shift;

      std::copy_n(obs     ,N*obs_dim,obs_buffer.data()      + next_index*obs_dim);
      std::copy_n(act     ,N*act_dim,act_buffer.data()      + next_index*act_dim);
      std::copy_n(rew     ,N        ,rew_buffer.data()      + next_index        );
      std::copy_n(next_obs,N*obs_dim,next_obs_buffer.data() + next_index*obs_dim);
      std::copy_n(done    ,N        ,done_buffer.data()     + next_index        );

      next_index += N;
      size = std::min(size+N,capacity);
    }

    template<typename T>
    void copy(const std::vector<T>& buffer,std::vector<T>& v,
	      std::size_t i,std::size_t dim) const {
      std::copy_n(buffer.data() + i*dim,dim,std::back_inserter(v));
    }

    template<typename T>
    void copy(const std::vector<T>& buffer,std::vector<std::vector<T>>& v,
	      std::size_t i,std::size_t dim) const {
      v.emplace_back(buffer.data() +  i   *dim,
		     buffer.data() + (i+1)*dim);
    }

  protected:
    std::mt19937 g;

    auto initialize_space(std::size_t size = 0ul) const {
      std::vector<std::vector<Observation>> obs{},next_obs{};
      std::vector<std::vector<Action>> act{};
      std::vector<Reward> rew{};
      std::vector<Done> done{};

      obs.reserve(size);
      act.reserve(size);
      rew.reserve(size);
      next_obs.reserve(size);
      done.reserve(size);

      return std::make_tuple(obs,act,rew,next_obs,done);
    }

    template<typename Obs_t,typename Act_t>
    void encode_sample(const std::vector<std::size_t>& indexes,
		       Obs_t& obs, Act_t& act,
		       std::vector<Reward>& rew,
		       Obs_t& next_obs,
		       std::vector<Done>& done) const {
      for(auto i : indexes){
	copy(obs_buffer     ,obs     ,i,obs_dim);
	copy(act_buffer     ,act     ,i,act_dim);
	copy(next_obs_buffer,next_obs,i,obs_dim);

	rew.push_back(rew_buffer[i]);
	done.push_back(done_buffer[i]);
      }
    }

    auto encode_sample(const std::vector<std::size_t>& indexes) const {
      auto [obs,act,rew,next_obs,done] = initialize_space(indexes.size());

      encode_sample(indexes,obs,act,rew,next_obs,done);
      return std::make_tuple(obs,act,rew,next_obs,done);
    }

  public:
    ReplayBuffer(std::size_t n,std::size_t obs_dim,std::size_t act_dim)
      : capacity(n),
	size{0},
	obs_dim{obs_dim},
	act_dim{act_dim},
	next_index{0ul},
	obs_buffer(capacity * obs_dim,Observation{0}),
	act_buffer(capacity * act_dim,Action{0}),
	rew_buffer(capacity,Reward{0}),
	next_obs_buffer(capacity * obs_dim,Observation{0}),
	done_buffer(capacity,Done{0}),
	g{std::random_device{}()} {}
    ReplayBuffer(): ReplayBuffer{1,1,1} {}
    ReplayBuffer(const ReplayBuffer&) = default;
    ReplayBuffer(ReplayBuffer&&) = default;
    ReplayBuffer& operator=(const ReplayBuffer&) = default;
    ReplayBuffer& operator=(ReplayBuffer&&) = default;
    ~ReplayBuffer() = default;

    std::size_t buffer_size() const { return size; }
    std::size_t get_next_index() const { return next_index;}
    std::size_t get_capacity() const { return capacity; }

    void add(Observation* obs,
	     Action* act,
	     Reward* rew,
	     Observation* next_obs,
	     Done* done,
	     std::size_t N = 1ul){

      auto copy_N = std::min(N,capacity - next_index);
      store_buffer(obs,act,rew,next_obs,done,0ul,copy_N);

      if(capacity == next_index){
	next_index = 0ul;
	store_buffer(obs,act,rew,next_obs,done,copy_N,N - copy_N);
      }
    }

    template<typename Obs_t,typename Act_t>
    void sample(std::size_t batch_size,
		Obs_t& obs, Act_t& act,
		std::vector<Reward>& rew,
		Obs_t& next_obs,
		std::vector<Done>& done){
      obs.resize(0);
      act.resize(0);
      rew.resize(0);
      next_obs.resize(0);
      done.resize(0);

      auto random = [this,d=rand_t{0,size-1}]()mutable{ return d(this->g); };
      auto indexes = std::vector<std::size_t>{};
      indexes.reserve(batch_size);
      std::generate_n(std::back_inserter(indexes),batch_size,random);

      encode_sample(indexes,obs,act,rew,next_obs,done);
    }

    auto sample(std::size_t batch_size){
      auto [obs,act,rew,next_obs,done] = initialize_space(batch_size);

      sample(batch_size,obs,act,rew,next_obs,done);

      return std::make_tuple(obs,act,rew,next_obs,done);
    }
  };

  template<typename Observation,typename Action,typename Reward,typename Done,
	   typename Priority>
  class PrioritizedReplayBuffer
    :public ReplayBuffer<Observation,Action,Reward,Done> {
  private:
    using BaseClass = ReplayBuffer<Observation,Action,Reward,Done>;
    Priority alpha;
    Priority max_priority;
    SegmentTree<Priority> sum;
    SegmentTree<Priority> min;

    void sample_proportional(std::size_t batch_size,
			     std::vector<std::size_t>& indexes){
      auto every_range_len
	= Priority{1.0} * sum.reduce(0,this->buffer_size()) / batch_size;

      std::generate_n(std::back_inserter(indexes),batch_size,
		      [=,i=0ul,
		       d=std::uniform_real_distribution<Priority>{}]()mutable{
			auto mass = (d(this->g) + (i++))*every_range_len;
			return this->sum.largest_region_index([=](auto v){
								return v <= mass;
							      },this->buffer_size());
		      });
    }

    auto sample_proportional(std::size_t batch_size){
      auto indexes = std::vector<std::size_t>{};
      indexes.reserve(batch_size);

      sample_proportional(batch_size,indexes);

      return indexes;
    }

    void set_weights(const std::vector<std::size_t>& indexes,Priority beta,
		     std::vector<Priority>& weights) const {
      auto b_size = this->buffer_size();
      auto inv_sum = Priority{1.0} / sum.reduce(0,b_size);
      auto p_min = min.reduce(0,b_size) * inv_sum;
      auto inv_max_weight = Priority{1.0} / std::pow(p_min * b_size,-beta);

      std::transform(indexes.begin(),indexes.end(),std::back_inserter(weights),
		     [=](auto idx){
		       auto p_sample = this->sum.get(idx) * inv_sum;
		       return std::pow(p_sample*b_size,-beta)*inv_max_weight;
		     });
    }

    auto set_weights(const std::vector<std::size_t>& indexes,Priority beta) const {
      std::vector<Priority> weights{};
      weights.reserve(indexes.size());

      set_weights(indexes,beta,weights);
      return weights;
    }

  public:
    PrioritizedReplayBuffer(std::size_t n,std::size_t obs_dim,std::size_t act_dim,
			    Priority alpha)
      : BaseClass{n,obs_dim,act_dim},
	alpha{std::max(alpha,Priority{0.0})},
	max_priority{1.0},
	sum{PowerOf2(n),[](auto a,auto b){ return a+b; }},
	min{PowerOf2(n),[zero = Priority{0}](Priority a,Priority b){
			  return ((zero == a) ? b:
				  (zero == b) ? a:
				  std::min(a,b));
			}} {}
    PrioritizedReplayBuffer() : PrioritizedReplayBuffer{1,1,1,0.0} {}
    PrioritizedReplayBuffer(const PrioritizedReplayBuffer&) = default;
    PrioritizedReplayBuffer(PrioritizedReplayBuffer&&) = default;
    PrioritizedReplayBuffer& operator=(const PrioritizedReplayBuffer&) = default;
    PrioritizedReplayBuffer& operator=(PrioritizedReplayBuffer&&) = default;
    ~PrioritizedReplayBuffer() = default;

    void add(Observation* obs,Action* act,Reward* rew,
	     Observation* next_obs,Done* done,std::size_t N = 1ul){
      auto next_idx = this->get_next_index();
      this->BaseClass::add(obs,act,rew,next_obs,done,N);

      auto v = std::pow(max_priority,alpha);
      sum.set(next_idx,v,N,this->buffer_size());
      min.set(next_idx,v,N,this->buffer_size());
    }

    template<typename Obs_t,typename Act_t>
    void sample(std::size_t batch_size,Priority beta,
		Obs_t& obs, Act_t& act,
		std::vector<Reward>& rew,
		Obs_t& next_obs,
		std::vector<Done>& done,
		std::vector<Priority>& weights,
		std::vector<std::size_t>& indexes){
      beta = std::max(beta,Priority{0});

      indexes.resize(0);
      indexes.reserve(batch_size);
      sample_proportional(batch_size,indexes);

      weights.resize(0);
      weights.reserve(batch_size);
      set_weights(indexes,beta,weights);

      this->BaseClass::encode_sample(indexes,obs,act,rew,next_obs,done);
    }

    template<typename Obs_t,typename Act_t>
    void sample(std::size_t batch_size,
		Obs_t& obs, Act_t& act,
		std::vector<Reward>& rew,
		Obs_t& next_obs,
		std::vector<Done>& done){
      std::vector<Priority> weights{};
      std::vector<std::size_t> indexes{};
      sample(batch_size,Priority{0.0},obs,act,rew,next_obs,done,weights,indexes);
    }

    auto sample(std::size_t batch_size,Priority beta){
      beta = std::max(beta,Priority{0});

      auto indexes = sample_proportional(batch_size);

      auto weights = set_weights(indexes,beta);

      auto samples = this->BaseClass::encode_sample(indexes);
      return std::tuple_cat(samples,std::make_tuple(weights,indexes));
    }

    auto sample(std::size_t batch_size){
      return sample(batch_size,Priority{0.0});
    }

    void update_priorities(std::vector<std::size_t>& indexes,
			   std::vector<Priority>& priorities){

      max_priority = std::accumulate(indexes.begin(),indexes.end(),max_priority,
				     [=,p=priorities.begin()]
				     (auto max_p, auto index) mutable {
				       auto v = std::pow(*p,this->alpha);
				       this->sum.set(index,v);
				       this->min.set(index,v);

				       return std::max(max_p,*(p++));
				     });
    }
  };
}
#endif // YMD_REPLAY_BUFFER_HH
