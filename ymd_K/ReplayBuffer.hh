#ifndef YMD_REPLAY_BUFFER_HH
#define YMD_REPLAY_BUFFER_HH 1

#include <cmath>
#include <vector>
#include <random>
#include <utility>
#include <deque>
#include <tuple>
#include <functional>

template<typename T> struct UnderlyingType { using type = T; };
template<typename T> struct UnderlyingType<std::vector<T>>{ using type = T; };

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

      auto random = [&g,d=rand_t{0,buffer.size()-1}] () mutable { return d(g); };

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

}
#endif // YMD_REPLAY_BUFFER_HH
