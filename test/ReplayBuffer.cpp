#include <iostream>
#include <chrono>
#include <tuple>

#include <ReplayBuffer.hh>

int main(){
  using Observation = double;
  using Action = double;
  using Reward = double;
  using Done = int;
  using Priority = double;

  constexpr const auto obs_dim = 3ul;
  constexpr const auto act_dim = 1ul;

  constexpr const auto N_step = 100ul;
  constexpr const auto N_buffer_size = 32ul;
  constexpr const auto N_batch_size = 16ul;

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

  auto dm = ymd::DimensionalBuffer<Observation>{N_buffer_size,obs_dim};
  auto v = std::vector<Observation>{};
  std::generate_n(std::back_inserter(v),obs_dim,
		  [i=0ul]()mutable{ return Observation(i++); });

  std::cout << "DimensionalBuffer: " << std::endl;
  Observation* obs_ptr = nullptr;
  dm.get_data(0ul,obs_ptr);
  std::cout << " DimensionalBuffer.data(): " << obs_ptr<< std::endl;
  std::cout << "*DimensionalBuffer.data(): " << *obs_ptr << std::endl;

  dm.store_data(v.data(),0ul,0ul,1ul);
  std::cout << " DimensionalBuffer[0]: " << obs_ptr[0] << std::endl;
  std::cout << "*DimensionalBuffer[1]: " << obs_ptr[1]  << std::endl;
  std::cout << " DimensionalBuffer[2]: " << obs_ptr[2] << std::endl;


  for(auto n = 0ul; n < N_times; ++n){
    auto next_index = std::min(n*obs_dim % N_buffer_size,N_buffer_size-1);
    dm.store_data(v.data(),0ul,next_index,1ul);
  }

  auto rb = ymd::ReplayBuffer<Observation,Action,Reward,Done>{N_buffer_size,
							      obs_dim,
							      act_dim};

  auto per = ymd::PrioritizedReplayBuffer<Observation,Action,
					  Reward,Done,Priority>{N_buffer_size,
								obs_dim,
								act_dim,
								alpha};


  for(auto i = 0ul; i < N_step; ++i){
    auto obs = std::vector<Observation>(obs_dim,0.1*i);
    auto act = std::vector<Action>(act_dim,2.0*i);
    auto rew = 0.1 * i;
    auto next_obs = std::vector<Observation>(obs_dim,0.1*(i+1));
    auto done = (N_step - 1 == i) ? 1: 0;

    rb.add(obs.data(),act.data(),&rew,next_obs.data(),&done);
    per.add(obs.data(),act.data(),&rew,next_obs.data(),&done);
  }

  rb.clear();
  per.clear();

  for(auto i = 0ul; i < N_step; ++i){
    auto obs = std::vector<Observation>(obs_dim,0.1*i);
    auto act = std::vector<Action>(act_dim,2.0*i);
    auto rew = 0.1 * i;
    auto next_obs = std::vector<Observation>(obs_dim,0.1*(i+1));
    auto done = (N_step - 1 == i) ? 1: 0;

    rb.add(obs.data(),act.data(),&rew,next_obs.data(),&done);
    per.add(obs.data(),act.data(),&rew,next_obs.data(),&done);
  }


  auto [rb_o,rb_a,rb_r,rb_no,rb_d] = rb.sample(N_batch_size);
  auto [per_o,per_a,per_r,per_no,per_d,per_w,per_i] = per.sample(N_batch_size,beta);

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
  timer([&](){ per.sample(N_batch_size,beta,
			  per_o,per_a,per_r,per_no,per_d,per_w,per_i); },N_times);

  return 0;
}
