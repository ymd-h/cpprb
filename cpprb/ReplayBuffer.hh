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
  template<typename T>
  class DimensionalBuffer {
  private:
    std::vector<T> buffer;
    std::size_t dim;
  public:
    DimensionalBuffer(std::size_t size,std::size_t dim)
      : buffer(size * dim,T{0}),
	dim{dim} {}
    DimensionalBuffer(): DimensionalBuffer{1ul,1ul}  {}
    DimensionalBuffer(const DimensionalBuffer&) = default;
    DimensionalBuffer(DimensionalBuffer&&) = default;
    DimensionalBuffer& operator=(const DimensionalBuffer&) = default;
    DimensionalBuffer& operator=(DimensionalBuffer&&) = default;
    virtual ~DimensionalBuffer() = default;
    void store_data(T* v,std::size_t shift,std::size_t next_index,std::size_t N){
      std::copy_n(v + shift*dim, N*dim,buffer.data() + next_index*dim);
    }
    void set_data(std::size_t ith,std::vector<T>& v) const {
      std::copy_n(buffer.data() + ith * dim, dim,std::back_inserter(v));
    }
    void set_data(std::size_t ith,std::vector<std::vector<T>>& v) const {
      v.emplace_back(buffer.data() +  ith    * dim,
		     buffer.data() + (ith+1) * dim);
    }
    void set_data(std::size_t ith,T*& v) const {
      v = (T*)(buffer.data()) + ith * dim;
    }
    void clear(){ buffer.resize(0); }
  };

  template<typename Observation,typename Action,typename Reward,typename Done>
  class InternalBuffer {
  private:
    const std::size_t buffer_size;
    std::size_t stored_size;
    std::size_t obs_dim;
    std::size_t act_dim;
    std::size_t next_index;
    DimensionalBuffer<Observation> obs_buffer;
    DimensionalBuffer<Action> act_buffer;
    DimensionalBuffer<Reward> rew_buffer;
    DimensionalBuffer<Observation> next_obs_buffer;
    DimensionalBuffer<Done> done_buffer;
    void store(Observation* obs, Action* act, Reward* rew,
	       Observation* next_obs, Done* done,
	       std::size_t shift, std::size_t N){
      obs_buffer     .store_data(     obs,shift,next_index,N);
      act_buffer     .store_data(     act,shift,next_index,N);
      rew_buffer     .store_data(     rew,shift,next_index,N);
      next_obs_buffer.store_data(next_obs,shift,next_index,N);
      done_buffer    .store_data(    done,shift,next_index,N);

      next_index += N;
      stored_size = std::min(stored_size+N,buffer_size);
    }

  public:
    InternalBuffer(std::size_t size,std::size_t obs_dim,std::size_t act_dim)
      : buffer_size{size},
	stored_size{0ul},
	obs_dim{obs_dim},
	act_dim{act_dim},
	next_index{0ul},
	obs_buffer{size,obs_dim},
	act_buffer{size,act_dim},
	rew_buffer{size,1ul},
	next_obs_buffer{size,obs_dim},
	done_buffer{size,1ul} {}
    InternalBuffer(): InternalBuffer{1ul,1ul,1ul} {}
    InternalBuffer(const InternalBuffer&) = default;
    InternalBuffer(InternalBuffer&&) = default;
    InternalBuffer& operator=(const InternalBuffer&) = default;
    InternalBuffer& operator=(InternalBuffer&&) = default;
    virtual ~InternalBuffer() = default;
    void store(Observation* obs, Action* act, Reward* rew,
	       Observation* next_obs, Done* done,
	       std::size_t N = 1ul){
      auto copy_N = std::min(N,buffer_size - next_index);
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
    void get(std::size_t index,Obs_t& obs,Act_t& act,Rew_t& rew,
	     Obs_t& next_obs,Done_t& done) const {
      obs_buffer     .set_data(index,     obs);
      act_buffer     .set_data(index,     act);
      rew_buffer     .set_data(index,     rew);
      next_obs_buffer.set_data(index,next_obs);
      done_buffer    .set_data(index,    done);
    }

    void get_buffer_pointers(Observation*& obs, Action*& act, Reward*& rew,
			     Observation*& next_obs, Done*& done) const {
      get(0ul,obs,act,rew,next_obs,done);
    }

    virtual void clear(){
      obs_buffer.clear();
      act_buffer.clear();
      rew_buffer.clear();
      next_obs_buffer.clear();
      done_buffer.clear();

      stored_size = 0ul;
      next_index = 0ul;
    }
  };

  template<typename Observation,typename Action,typename Reward,typename Done>
  class ReplayBuffer : public InternalBuffer<Observation,Action,Reward,Done>{
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
      this->Buffer_t::store(obs,act,rew,next_obs,done,N);
    }

    template<typename Obs_t,typename Act_t,typename  Rew_t,typename Done_t>
    void sample(std::size_t batch_size,
		Obs_t& obs, Act_t& act,
		Rew_t& rew, Obs_t& next_obs, Done_t& done,
		std::vector<size_t>& indexes){
      auto random = [this,
		     d=rand_t{0,this->Buffer_t::get_stored_size()-1}]()mutable{
		      return d(this->g);
		    };
      indexes.resize(0);
      indexes.reserve(batch_size);
      std::generate_n(std::back_inserter(indexes),batch_size,random);

      encode_sample(indexes,obs,act,rew,next_obs,done);
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

  template<typename Priority>
  class PrioritizedSampler {
  private:
    Priority alpha;
    Priority max_priority;
    const Priority default_max_priority;
    SegmentTree<Priority> sum;
    SegmentTree<Priority> min;
    std::mt19937 g;

    void sample_proportional(std::size_t batch_size,
			     std::vector<std::size_t>& indexes,
			     std::size_t stored_size){
      indexes.resize(0);
      indexes.reserve(batch_size);

      auto every_range_len
	= Priority{1.0} * sum.reduce(0,stored_size) / batch_size;

      std::generate_n(std::back_inserter(indexes),batch_size,
		      [=,i=0ul,
		       d=std::uniform_real_distribution<Priority>{}]()mutable{
			auto mass = (d(this->g) + (i++))*every_range_len;
			return this->sum.largest_region_index([=](auto v){
								return v <= mass;
							      },stored_size);
		      });
    }

    void set_weights(const std::vector<std::size_t>& indexes,Priority beta,
		     std::vector<Priority>& weights,std::size_t stored_size) const {
      auto b_size = stored_size;
      auto inv_sum = Priority{1.0} / sum.reduce(0,b_size);
      auto p_min = min.reduce(0,b_size) * inv_sum;
      auto inv_max_weight = Priority{1.0} / std::pow(p_min * b_size,-beta);

      std::transform(indexes.begin(),indexes.end(),std::back_inserter(weights),
		     [=](auto idx){
		       auto p_sample = this->sum.get(idx) * inv_sum;
		       return std::pow(p_sample*b_size,-beta)*inv_max_weight;
		     });
    }

    template<typename F>
    void set_priorities(std::size_t next_index,F&& f,
			std::size_t N,std::size_t stored_size){
      sum.set(next_index,std::forward<F>(f),N,stored_size);
      min.set(next_index,std::forward<F>(f),N,stored_size);
    }

  public:
    PrioritizedSampler(std::size_t buffer_size,Priority alpha)
      : alpha{alpha},
	max_priority{1.0},
	default_max_priority{1.0},
	sum{PowerOf2(buffer_size),[](auto a,auto b){ return a+b; }},
	min{PowerOf2(buffer_size),[zero = Priority{0}](Priority a,
						       Priority b){
				    return ((zero == a) ? b:
					    (zero == b) ? a:
					    std::min(a,b));
				  }},
	g{std::random_device{}()} {}
    PrioritizedSampler() = default;
    PrioritizedSampler(const PrioritizedSampler&) = default;
    PrioritizedSampler(PrioritizedSampler&&) = default;
    PrioritizedSampler& operator=(const PrioritizedSampler&) = default;
    PrioritizedSampler& operator=(PrioritizedSampler&&) = default;
    ~PrioritizedSampler() = default;

    void sample(std::size_t batch_size,Priority beta,
		std::vector<Priority>& weights,std::vector<std::size_t>& indexes,
		std::size_t stored_size){
      sample_proportional(batch_size,indexes,stored_size);
      set_weights(indexes,beta,weights,stored_size);
    }
    virtual void clear(){
      max_priority = default_max_priority;
    }

    Priority get_max_priority() const {
      return max_priority;
    }

    void set_priorities(std::size_t next_index,Priority p){
      auto v = std::pow(p,alpha);
      sum.set(next_index,v);
      min.set(next_index,v);
    }

    void set_priorities(std::size_t next_index){
      set_priorities(next_index,max_priority);
    }

    void set_priorities(std::size_t next_index,Priority* p,
			std::size_t N,std::size_t stored_size){
      set_priorities(next_index,[=]() mutable { return std::pow(*(p++),alpha); },
		     N,stored_size);
    }

    void set_priorities(std::size_t next_index,
			std::size_t N,std::size_t stored_size){
      set_priorities(next_index,[=](){ return std::pow(max_priority,alpha); },
		     N,stored_size);
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

  template<typename Observation,typename Action,typename Reward,typename Done,
	   typename Priority>
  class PrioritizedReplayBuffer:
    public ReplayBuffer<Observation,Action,Reward,Done>,
    public PrioritizedSampler<Priority> {
  private:
    using BaseClass = ReplayBuffer<Observation,Action,Reward,Done>;
    using Sampler = PrioritizedSampler<Priority>;
  public:
    PrioritizedReplayBuffer(std::size_t n,std::size_t obs_dim,std::size_t act_dim,
			    Priority alpha)
      : BaseClass{n,obs_dim,act_dim},
	Sampler{n,alpha} {}
    PrioritizedReplayBuffer() : PrioritizedReplayBuffer{1,1,1,0.0} {}
    PrioritizedReplayBuffer(const PrioritizedReplayBuffer&) = default;
    PrioritizedReplayBuffer(PrioritizedReplayBuffer&&) = default;
    PrioritizedReplayBuffer& operator=(const PrioritizedReplayBuffer&) = default;
    PrioritizedReplayBuffer& operator=(PrioritizedReplayBuffer&&) = default;
    virtual ~PrioritizedReplayBuffer() override = default;

    virtual void add(Observation* obs,Action* act,Reward* rew,
		     Observation* next_obs,Done* done,std::size_t N) override {
      auto next_index = this->get_next_index();
      this->BaseClass::add(obs,act,rew,next_obs,done,N);
      this->set_priorities(next_index,N,this->get_stored_size());
    }

    virtual void add(Observation* obs,Action* act,Reward* rew,
		     Observation* next_obs,Done* done,
		     Priority* priority,std::size_t N){
      auto next_index = this->get_next_index();
      this->BaseClass::add(obs,act,rew,next_obs,done,N);
      this->set_priorities(next_index,priority,N,this->get_stored_size());
    }

    virtual void add(Observation* obs,Action* act,Reward* rew,
		     Observation* next_obs,Done* done,Priority p){
      auto next_index = this->get_next_index();
      this->BaseClass::add(obs,act,rew,next_obs,done,1ul);
      this->set_priorities(next_index,p);
    }

    virtual void add(Observation* obs,Action* act,Reward* rew,
		     Observation* next_obs,Done* done){
      auto next_index = this->get_next_index();
      this->BaseClass::add(obs,act,rew,next_obs,done,1ul);
      this->set_priorities(next_index);
    }

    template<typename Obs_t,typename Act_t,typename Rew_t,typename Done_t>
    void sample(std::size_t batch_size,Priority beta,
		Obs_t& obs, Act_t& act,
		Rew_t& rew, Obs_t& next_obs, Done_t& done,
		std::vector<Priority>& weights,
		std::vector<std::size_t>& indexes){

      this->Sampler::sample(batch_size,beta,weights,indexes,this->get_stored_size());

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

      std::vector<std::size_t> indexes{};
      indexes.reserve(batch_size);

      std::vector<Priority> weights{};
      weights.reserve(batch_size);

      this->Sampler::sample(batch_size,beta,weights,indexes,this->get_stored_size());

      auto samples = this->BaseClass::encode_sample(indexes);
      return std::tuple_cat(samples,std::make_tuple(weights,indexes));
    }

    auto sample(std::size_t batch_size){
      return sample(batch_size,Priority{0.0});
    }

    virtual void clear() override {
      this->BaseClass::clear();
      this->Sampler::clear();
    }
  };
}
#endif // YMD_REPLAY_BUFFER_HH
