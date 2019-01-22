#include <iostream>
#include <chrono>
#include <tuple>

#include <ReplayBuffer.hh>

int main(){
  using Observation = std::vector<double>;
  using Action = std::vector<double>;
  using Reward = double;
  using Done = int;
  using Priority = double;

  constexpr const auto obs_dim = 3ul;
  constexpr const auto act_dim = 1ul;

  constexpr const auto N_step = 100ul;
  constexpr const auto N_buffer_size = 15ul;
  constexpr const auto N_batch_size = 256ul;

  constexpr const auto N_times = 1000ul;

  auto timer = [](auto&& f,auto N){
		 auto start = std::chrono::high_resolution_clock::now();

		 for(auto i = 0ul; i < N; ++i){ f(); }

		 auto end = std::chrono::high_resolution_clock::now();
		 auto elapsed = end - start;

		 auto s = std::chrono::duration_cast<std::chrono::seconds>(elapsed);
		 auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(elapsed);
		 auto us = std::chrono::duration_cast<std::chrono::microseconds>(elapsed);
		 auto ns = std::chrono::duration_cast<std::chrono::nanoseconds>(elapsed);
		 std::cout << s.count() << "s "
			   << ms.count() - s.count() * 1000 << "ms "
			   << us.count() - ms.count() * 1000 << "us "
			   << ns.count() - us.count() * 1000 << "ns"
			   << std::endl;
	       };

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
    auto done = (N_step - 1 == i) ? 1: 0;

    rb.add(obs,act,rew,next_obs,done);
    per.add(obs,act,rew,next_obs,done);
  }

  auto [rb_o,rb_a,rb_r,rb_no,rb_d] = rb.sample(N_batch_size);
  auto [per_o,per_a,per_r,per_no,per_d,per_w,per_i] = per.sample(N_batch_size,beta);

  auto show_vector = [](auto v,auto name){
		       std::cout << name << ": ";
		       for(auto ve: v){ std::cout << ve << " "; }
		       std::cout << std::endl;
		     };

  auto show_vector_of_vector = [](auto v,std::string name){
				 std::cout << name << ": " << std::endl;
				 for(auto ve: v){
				   std::cout << " ";
				   for(auto vee: ve){ std::cout << vee << " "; }
				   std::cout << std::endl;
				 }
				 std::cout << std::endl;
			       };

  std::cout << "ReplayBuffer" << std::endl;
  show_vector_of_vector(rb_o,"obs");
  show_vector_of_vector(rb_a,"act");
  show_vector(rb_r,"rew");
  show_vector_of_vector(rb_no,"next_obs");
  show_vector(rb_d,"done");

  std::cout << std::endl;

  std::cout << "PrioritizedReplayBuffer" << std::endl;
  show_vector_of_vector(per_o,"obs");
  show_vector_of_vector(per_a,"act");
  show_vector(per_r,"rew");
  show_vector_of_vector(per_no,"next_obs");
  show_vector(per_d,"done");
  show_vector(per_w,"weights");
  show_vector(per_i,"indexes");

  per.update_priorities(per_i,per_w);

  rb.sample(N_batch_size,rb_o,rb_a,rb_r,rb_no,rb_d);

  per.sample(N_batch_size,beta,per_o,per_a,per_r,per_no,per_d,per_w,per_i);

  std::cout << std::endl;
  std::cout << "PER Sample: " << N_times << " times execution" << std::endl;
  timer([&](){ per.sample(N_batch_size,beta,per_o,per_a,per_r,per_no,per_d,per_w,per_i); },N_times);

  return 0;
}
