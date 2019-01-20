#include <iostream>

#include <ReplayBuffer.hh>

int main(){
  using Observation = std::vector<double>;
  using Action = std::vector<double>;
  using Reward = double;
  using Done = int;
  using Priority = double;

  constexpr const auto obs_dim = 15ul;
  constexpr const auto act_dim = 5ul;

  constexpr const auto N_step = 100ul;
  constexpr const auto N_buffer_size = 15ul;
  constexpr const auto N_batch_size = 5ul;

  auto alpha = 0.7;
  auto beta = 0.5;

  auto rb = ymd::ReplayBuffer<Observation,Action,Reward,Done>{N_buffer_size};

  auto per = ymd::PrioritizedReplayBuffer<Observation,Action,
					  Reward,Done,Priority>{N_buffer_size,alpha};


  for(auto i = 0ul; i < N_step; ++i){
    auto obs = Observation(obs_dim,0.1*i);
    auto act = Action(act_dim,2.0*i);
    auto rew = 0.1 * i;
    auto next_obs = Observation(obs_dim,0.1*(i+1));
    auto done = (N_step - 1 == i) ? 0: 1;

    rb.add(obs,act,rew,next_obs,done);
    per.add(obs,act,rew,next_obs,done);
  }

  auto rb_sample = rb.sample(N_batch_size);
  auto per_sample =per.sample(N_batch_size);

  return 0;
}
