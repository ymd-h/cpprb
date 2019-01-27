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
  class InternalBuffer {
  private:
    const std::size_t baffer_size;
    std::size_t stored_size;
    std::size_t obs_dim;
    std::size_t act_dim;
    std::size_t next_index;
    std::vector<Observation> obs_buffer;
    std::vector<Action> act_buffer;
    std::vector<Reward> rew_buffer;
    std::vector<Observation> next_obs_buffer;
    std::vector<Done> done_buffer;
    template<typename T>
    void store_data(T* v,std::size_t shift,std::size_t dim,std::size_t N,
		    std::vector<T>& buffer){
      std::copy_n(v + shift*dim, N*dim,buffer.data() + next_index*dim);
    }
    void store(Observation* obs, Action* act, Reward* rew,
	       Observation* next_obs, Done* done,
	       std::size_t shift, std::size_t N){
      store_data(     obs,shift,obs_dim,N,     obs_buffer);
      store_data(     act,shift,act_dim,N,     act_buffer);
      store_data(     rew,shift,    1ul,N,     rew_buffer);
      store_data(next_obs,shift,obs_dim,N,next_obs_buffer);
      store_data(    done,shift,    1ul,N,    done_buffer);

      next_index += N;
      stored_size = std::min(stored_size+N,buffer_size);
    }

    template<typename T>
    void set_data(const std::vector<T>& buffer, std::size_t ith,
		  std::size_t dim,std::vector<T>& v) const {
      std::copy_n(buffer.data() + ith * dim, dim,std::back_inserter(v));
    }

    template<typename T>
    void set_data(const std::vector<T>& buffer, std::size_t ith,
		  std::size_t dim,std::vector<std::vector<T>>& v) const {
      v.emplace_back(buffer.data() +  ith    * dim,
		     buffer.data() + (ith+1) * dim);
    }

    template<typename T>
    void set_data(const std::vector<T>& buffer, std::size_t ith,
		  std::size_t dim,T*& v) const {
      v = buffer.data() + ith * dim
    }

  public:
    InternalBuffer(std::size_t size,std::size_t obs_dim,std::size_t act_dim)
      : buffer_size{size},
	stored_size{0ul},
	obs_dim{obs_dim},
	act_dim{act_dim},
	next_index{0ul},
	obs_buffer(size * obs_dim,Observation{0}),
	act_buffer(size * act_dim,Action{0}),
	rew_buffer(size,Reward{0}),
	next_obs_buffer(size * obs_dim,Observation{0}),
	done_buffer(size,Done{0}) {}
    InternalBuffer(): InternalBuffer{1ul,1ul,1ul} {}
    InternalBuffer(const InternalBuffer&) = default;
    InternalBuffer(InternalBuffer&&) = default;
    InternalBuffer& operator=(const InternalBuffer&) = default;
    InternalBuffer& operator=(InternalBuffer&&) = default;
    virtual ~InternalBuffer() = default;
    void store(Observation* obs, Action* act, Reward* rew,
	       Observation* next_obs, Done* done,
	       std::size_t N = 1ul){
      auto copy_N = std::min(N,capacity - next_index);
      store(obs,act,rew,next_obs,done,0ul,copy_N);

      if(buffer_size == next_index){
	next_index = 0ul;
	store(obs,act,rew,next_obs,done,copy_N,N - copy_N);
      }
    }

    std::size_t get_buffer_size() const { return buffer_size; }
    std::size_t get_stored_size() const { return stored_size; }
    std::size_t get_next_index() const { return next_index; }

    template<typename Obs_t,typename Act_t,typename Rew_t,typename Done_t>
    void get(std::size_t index,Obs_t& obs,Act_t& act,Rew_t& rew,Done_t& done){
      set_data(     obs_buffer,index,obs_dim,     obs);
      set_data(     act_buffer,index,act_dim,     act);
      set_data(     rew_buffer,index,    1ul,     rew);
      set_data(next_obs_buffer,index,obs_dim,next_obs);
      set_data(    done_buffer,index,    1ul,    done);
    }

    void get_buffer_pointers(Observation*& obs, Action*& act, Reward*& rew,
			     Observation*& next_obs, Done*& done){
      get(0ul,obs,act,rew,next_obs,done);
    }

    virtual void clear(){
      obs_buffer.resize(0);
      act_buffer.resize(0);
      rew_buffer.resize(0);
      next_obs_buffer.resize(0);
      done_buffer.resize(0);

      stored_size = 0ul;
      next_index = 0ul;
    }
  };

  template<typename Observation,typename Action,typename Reward,typename Done>
  class ReplayBuffer : InternalBuffer<Observation,Action,Reward,Done>{
  public:
    using Buffer_t = InternalBuffer<Observation,Action,Reward,Done>;
    using rand_t = std::uniform_int_distribution<std::size_t>;
  private:
    std::vector<std::size_t> index_buffer;

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
      obs.resize(0);
      act.resize(0);
      rew.resize(0);
      next_obs.resize(0);
      done.resize(0);

      for(auto i : indexes){
	this->get(i,obs,act,rew,next_obs,done);
      }
    }

    auto encode_sample(const std::vector<std::size_t>& indexes) const {
      auto [obs,act,rew,next_obs,done] = initialize_space(indexes.size());

      encode_sample(indexes,obs,act,rew,next_obs,done);
      return std::make_tuple(obs,act,rew,next_obs,done);
    }

  public:
    ReplayBuffer(std::size_t n,std::size_t obs_dim,std::size_t act_dim)
      : Buffer_t{n,obs_dim,act_dim},
	index_buffer{},
	g{std::random_device{}()} {}
    ReplayBuffer(Buffer_t&& buffer)
      : Buffer_t{buffer},
	index_buffer{},
	g{std::random_device{}()} {}
    ReplayBuffer(): ReplayBuffer{1,1,1} {}
    ReplayBuffer(const ReplayBuffer&) = default;
    ReplayBuffer(ReplayBuffer&&) = default;
    ReplayBuffer& operator=(const ReplayBuffer&) = default;
    ReplayBuffer& operator=(ReplayBuffer&&) = default;
    virtual ~ReplayBuffer() = default;

    virtual void add(Observation* obs,
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

    template<typename Obs_t,typename Act_t,typename  Rew_t,typename Done_t>
    void sample(std::size_t batch_size,
		Obs_t& obs, Act_t& act,
		Rew_t& rew, Obs_t& next_obs, Done_t& done,
		std::vector<size_t>& indexes){
      auto random = [this,d=rand_t{0,size-1}]()mutable{ return d(this->g); };
      indexes.resize(0);
      indexes.reserve(batch_size);
      std::generate_n(std::back_inserter(indexes),batch_size,random);

      set_data(indexes,obs,act,rew,next_obs,done);
    }

    template<typename Obs_t,typename Act_t,typename  Rew_t,typename Done_t>
    void sample(std::size_t batch_size,
		Obs_t& obs, Act_t& act,
		Rew_t& rew, Obs_t& next_obs, Done_t& done){
      sample(batch_size,obs,act,rew,next_obs,done,index_buffer);
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
    Priority default_max_priority;
    SegmentTree<Priority> sum;
    SegmentTree<Priority> min;

    void sample_proportional(std::size_t batch_size,
			     std::vector<std::size_t>& indexes){
      indexes.resize(0);
      indexes.reserve(batch_size);

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

    template<typename F>
    void multi_add(Observation* obs,Action* act,Reward* rew,
		   Observation* next_obs,Done* done,F&& f,std::size_t N){
      auto next_idx = this->get_next_index();
      this->BaseClass::add(obs,act,rew,next_obs,done,N);

      sum.set(next_idx,std::forward<F>(f),N,this->buffer_size());
      min.set(next_idx,std::forward<F>(f),N,this->buffer_size());
    }

  public:
    PrioritizedReplayBuffer(std::size_t n,std::size_t obs_dim,std::size_t act_dim,
			    Priority alpha)
      : BaseClass{n,obs_dim,act_dim},
	alpha{std::max(alpha,Priority{0.0})},
	max_priority{1.0},
	default_max_priority{1.0},
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
    virtual ~PrioritizedReplayBuffer() override = default;

    virtual void add(Observation* obs,Action* act,Reward* rew,
		     Observation* next_obs,Done* done,std::size_t N) override {
      multi_add(obs,act,rew,next_obs,done,
		[v=std::pow(max_priority,alpha)](){ return v; },N);
    }

    virtual void add(Observation* obs,Action* act,Reward* rew,
		     Observation* next_obs,Done* done,
		     Priority* priority,std::size_t N){
      multi_add(obs,act,rew,next_obs,done,
		[=]()mutable{ return std::pow(*(priority++),this->alpha); },N);
    }

    virtual void add(Observation* obs,Action* act,Reward* rew,
		     Observation* next_obs,Done* done,Priority p){
      auto next_idx = this->get_next_index();
      this->BaseClass::add(obs,act,rew,next_obs,done,1ul);

      auto v = std::pow(p,alpha);
      sum.set(next_idx,v);
      min.set(next_idx,v);
    }

    virtual void add(Observation* obs,Action* act,Reward* rew,
		     Observation* next_obs,Done* done){
      add(obs,act,rew,next_obs,done,max_priority);
    }

    void prioritized_indexes(std::size_t batch_size,Priority beta,
			     std::vector<Priority>& weights,
			     std::vector<std::vector>& indexes){
      beta = std::max(beta,Priority{0});

      indexes.resize(0);
      indexes.reserve(batch_size);
      sample_proportional(batch_size,indexes);

      weights.resize(0);
      weights.reserve(batch_size);
      set_weights(indexes,beta,weights);
    }

    template<typename Obs_t,typename Act_t,typename Rew_t,typename Done_t>
    void sample(std::size_t batch_size,Priority beta,
		Obs_t& obs, Act_t& act,
		Rew_t& rew, Obs_t& next_obs, Done_t& done,
		std::vector<Priority>& weights,
		std::vector<std::size_t>& indexes){

      prioritized_indexes(batch_size,beta,weights,indexes);

      this->BaseClass::set_data(indexes,obs,act,rew,next_obs,done);
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

    virtual void clear() override {
      this->BaseClass::clear();
      max_priority = default_max_priority;
    }

    Priority get_max_priority() const {
      return max_priority;
    }
  };
}
#endif // YMD_REPLAY_BUFFER_HH
