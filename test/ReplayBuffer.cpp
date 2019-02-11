#include <iostream>
#include <chrono>
#include <tuple>
#include <string>
#include <cassert>
#include <cmath>
#include <type_traits>
#include <future>
#include <thread>

#include <ReplayBuffer.hh>

using namespace std::literals;

using Observation = double;
using Action = double;
using Reward = double;
using Done = double;
using Priority = double;

template<typename F>
inline auto timer(F&& f,std::size_t N){
  auto start = std::chrono::high_resolution_clock::now();

  for(std::size_t i = 0ul; i < N; ++i){ f(); }

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
}

template<typename T>
void show_vector(T v,std::string name){
  std::cout << name << ": ";
  for(auto ve: v){ std::cout << ve << " "; }
  std::cout << std::endl;
}

template<typename T>
void show_vector_of_vector(T v,std::string name){
  std::cout << name << ": " << std::endl;
  for(auto ve: v){
    std::cout << " ";
    for(auto vee: ve){ std::cout << vee << " "; }
    std::cout << std::endl;
  }
  std::cout << std::endl;
}

template<typename T>
void show_pointer(T ptr,std::size_t N,std::string name){
  auto v = std::vector<std::remove_pointer_t<T>>(ptr,ptr+N);
  show_vector(v,name);
}

void test_NstepReward(){
  constexpr const auto buffer_size = 16ul;
  constexpr const auto obs_dim = 3ul;
  constexpr const auto nstep = 4;
  constexpr const auto gamma = 0.99;

  auto rb = ymd::CppNstepRewardBuffer<Observation,Reward>(buffer_size,
							  obs_dim,nstep,gamma);

  auto rew = std::vector(buffer_size,Reward{1});
  auto next_obs = std::vector(buffer_size * obs_dim,Observation{0});
  std::iota(next_obs.begin(),next_obs.end(),Observation{1});

  auto done = std::vector(buffer_size,Done{0});
  done.back() = Done{1};
  done[buffer_size / 2] = Done{1};

  auto indexes = std::vector(buffer_size,0ul);
  std::iota(indexes.begin(),indexes.end(),0ul);

  Reward *discounts, *ret;
  Observation* nstep_next_obs;

  rb.sample(indexes,rew.data(),next_obs.data(),done.data());
  rb.get_buffer_pointers(discounts,ret,nstep_next_obs);

  std::cout << std::endl
	    << "NstepRewardBuffer "
	    << "(buffer_size=" << buffer_size
	    << ",nstep=" << nstep
	    << ",gamma=" << gamma
	    << ")" << std::endl;

  std::cout << "[Input]" << std::endl;
  show_vector(rew,"rew");
  show_vector(next_obs,"next_obs (obs_dim="s + std::to_string(obs_dim) + ")");
  show_vector(done,"done");

  std::cout << "[Output]" << std::endl;
  show_pointer(discounts,buffer_size,"discounts");
  show_pointer(ret,buffer_size,"ret");
  show_pointer(nstep_next_obs,buffer_size*obs_dim,"nstep_next_obs");

  auto r =std::vector<Reward>{};
  std::generate_n(std::back_inserter(r),nstep,
		  [=,i=0ul]() mutable { return std::pow(gamma,i++); });

  for(auto i = 0ul; i < buffer_size; ++i){
    std::size_t end = std::distance(done.begin(),
				    std::find_if(done.begin()+i,done.end(),
						 [last=0.0](auto v) mutable {
						   return std::exchange(last,v) > 0.5;
						 }));
    auto exp_d = (i + nstep -1 < end ? r.back(): r[end - i -1]);
    if(std::abs(discounts[i] - exp_d) > exp_d * 0.001){
      std::cout << "discounts["<< i << "] != " << exp_d << std::endl;
      assert(!(std::abs(discounts[i] - exp_d) > exp_d * 0.001));
    }
  }
}

void test_SelectiveEnvironment(){
  constexpr const auto obs_dim = 3ul;
  constexpr const auto act_dim = 1ul;
  constexpr const auto episode_len = 4ul;
  constexpr const auto Nepisodes = 10ul;

  auto se = ymd::CppSelectiveEnvironment<Observation,Action,Reward,Done>(episode_len,
									 Nepisodes,
									 obs_dim,
									 act_dim);

  std::cout << std::endl
	    << "SelectiveEnvironment("
	    << "episode_len=" << episode_len
	    << ",Nepisodes=" << Nepisodes
	    << ",obs_dim=" << obs_dim
	    << ",act_dim=" << act_dim
	    << ")" << std::endl;

  assert(0ul == se.get_next_index());
  assert(0ul == se.get_stored_size());
  assert(0ul == se.get_stored_episode_size());
  assert(episode_len*Nepisodes == se.get_buffer_size());

  auto obs = std::vector(obs_dim*(episode_len+1),Observation{1});
  auto act = std::vector(act_dim*episode_len,Action{1.5});
  auto rew = std::vector(episode_len,Reward{1});
  auto done = std::vector(episode_len,Done{0});
  done.back() = Done{1};

  // Add 1-step
  se.store(obs.data(),act.data(),rew.data(),obs.data()+1,done.data(),1ul);
  auto [obs_,act_,rew_,next_obs_,done_,ep_len] = se.get_episode(0);
  show_pointer(obs_,se.get_stored_size()*obs_dim,"obs");
  show_pointer(act_,se.get_stored_size()*act_dim,"act");
  show_pointer(rew_,se.get_stored_size(),"rew");
  show_pointer(next_obs_,se.get_stored_size()*obs_dim,"next_obs");
  show_pointer(done_,se.get_stored_size(),"done");

  assert(1ul == ep_len);
  assert(1ul == se.get_next_index());
  assert(1ul == se.get_stored_size());
  assert(1ul == se.get_stored_episode_size());

  // Add remained 3-steps
  se.store(obs.data()+1,act.data()+1,rew.data()+1,obs.data()+2,done.data()+1,
	   episode_len - 1ul);
  se.get_episode(0,ep_len,obs_,act_,rew_,next_obs_,done_);
  show_pointer(obs_,se.get_stored_size()*obs_dim,"obs");
  show_pointer(act_,se.get_stored_size()*act_dim,"act");
  show_pointer(rew_,se.get_stored_size(),"rew");
  show_pointer(next_obs_,se.get_stored_size()*obs_dim,"next_obs");
  show_pointer(done_,se.get_stored_size(),"done");

  assert(episode_len == ep_len);
  assert(episode_len == se.get_next_index());
  assert(episode_len == se.get_stored_size());
  assert(1ul == se.get_stored_episode_size());

  // Try to get non stored episode
  se.get_episode(1,ep_len,obs_,act_,rew_,next_obs_,done_);
  assert(0ul == ep_len);

  // Add shorter epsode
  se.store(obs.data()+1,act.data()+1,rew.data()+1,obs.data()+2,done.data()+1,
	   episode_len - 1ul);
  se.get_episode(0,ep_len,obs_,act_,rew_,next_obs_,done_);
  show_pointer(obs_,se.get_stored_size()*obs_dim,"obs");
  show_pointer(act_,se.get_stored_size()*act_dim,"act");
  show_pointer(rew_,se.get_stored_size(),"rew");
  show_pointer(next_obs_,se.get_stored_size()*obs_dim,"next_obs");
  show_pointer(done_,se.get_stored_size(),"done");

  assert(2*episode_len - 1ul == se.get_next_index());
  assert(2*episode_len - 1ul == se.get_stored_size());
  assert(2ul == se.get_stored_episode_size());

  se.get_episode(1,ep_len,obs_,act_,rew_,next_obs_,done_);
  assert(episode_len - 1ul == ep_len);

  // Delete non existing episode
  assert(0ul == se.delete_episode(99));
  assert(2*episode_len - 1ul == se.get_next_index());
  assert(2*episode_len - 1ul == se.get_stored_size());
  assert(2ul == se.get_stored_episode_size());

  // Delete 0
  se.delete_episode(0);
  se.get_episode(0,ep_len,obs_,act_,rew_,next_obs_,done_);
  show_pointer(obs_,se.get_stored_size()*obs_dim,"obs");
  show_pointer(act_,se.get_stored_size()*act_dim,"act");
  show_pointer(rew_,se.get_stored_size(),"rew");
  show_pointer(next_obs_,se.get_stored_size()*obs_dim,"next_obs");
  show_pointer(done_,se.get_stored_size(),"done");
  assert(episode_len - 1ul == se.get_next_index());
  assert(episode_len - 1ul == se.get_stored_size());
  assert(1ul == se.get_stored_episode_size());

  // Add shorter epsode with not terminating
  se.store(obs.data(),act.data(),rew.data(),obs.data()+1,done.data(),
	   episode_len - 1ul);
  assert(2*episode_len - 2ul == se.get_next_index());
  assert(2*episode_len - 2ul == se.get_stored_size());
  assert(2ul == se.get_stored_episode_size());

  // Delete half-open episode
  se.delete_episode(1);
  assert(episode_len - 1ul == se.get_next_index());
  assert(episode_len - 1ul == se.get_stored_size());
  assert(1ul == se.get_stored_episode_size());

  // Add shorter epsode with not terminating
  se.store(obs.data(),act.data(),rew.data(),obs.data()+1,done.data(),
	   episode_len - 1ul);
  assert(2*episode_len - 2ul == se.get_next_index());
  assert(2*episode_len - 2ul == se.get_stored_size());
  assert(2ul == se.get_stored_episode_size());

  // Delete 0 when finishing half-open episode
  se.delete_episode(0);
  assert(episode_len - 1ul == se.get_next_index());
  assert(episode_len - 1ul == se.get_stored_size());
  assert(1ul == se.get_stored_episode_size());
}

void test_MultiThreadRingEnvironment(){
  constexpr const auto buffer_size = 1024ul * 256ul;
  constexpr const auto obs_dim = 3ul;
  constexpr const auto act_dim = 1ul;
  constexpr const auto add_dim = 100ul;
  constexpr const auto N_step = buffer_size * 3;
  constexpr const auto N_times = 10;

  std::cout << std::endl << "Multi-Thread RingEnvironment("
	    << "buffer_size=" << buffer_size
	    << ",obs_dim=" << obs_dim
	    << ",act_dim=" << act_dim
	    << ")" << std::endl;

  using NoLock_t = ymd::CppRingEnvironment<Observation,Action,Reward,Done,false>;
  using Lock_t = ymd::CppThreadSafeRingEnvironment<Observation,Action,Reward,Done>;

  auto single = NoLock_t(buffer_size,obs_dim,act_dim);
  auto multi  = Lock_t(buffer_size,obs_dim,act_dim);
  auto multi2 = Lock_t(buffer_size,obs_dim,act_dim);


  auto obs = std::vector<Observation>{};
  auto act = std::vector<Action>{};
  auto rew = std::vector<Reward>{};
  auto next_obs = std::vector<Observation>{};
  auto done = std::vector<Done>{};

  obs.reserve(obs_dim * buffer_size);
  act.reserve(act_dim * buffer_size);
  rew.reserve(buffer_size);
  next_obs.reserve(obs_dim * buffer_size);
  done.reserve(buffer_size);

  for(auto i = 0ul; i < N_step; ++i){
    std::fill_n(std::back_inserter(obs),obs_dim,1.0*i);
    std::fill_n(std::back_inserter(act),act_dim,0.5*i);
    rew.push_back(0.1*i);
    std::fill_n(std::back_inserter(next_obs),obs_dim,1.0*(i+1));
    done.push_back(i % 25 == 0 ? 1: 0);
  }

  std::cout << N_times << " times execution" << std::endl;

  auto f_core =
    [&](auto& b, std::size_t N_add,std::size_t start,std::size_t stop){
      for(auto i = start; i < stop; i += N_add){
	b.store(obs.data() + i*obs_dim,
		act.data() + i*act_dim,
		rew.data() + i,
		next_obs.data() + i*obs_dim,
		done.data() + i,
		std::min(N_add,stop-i));
      }
    };

  auto f =
    [&](auto& b,std::size_t N_add){
      std::cout << "Adding " << N_add << " time-points at once" << std::endl;
      timer([&]() mutable {
	      f_core(b,N_add,0ul,N_step);
	    },N_times);
    };

  auto multi_f =
    [&](auto& b,std::size_t N_add,auto cores){

      const std::size_t n = N_step/cores;

      auto v = std::vector<std::future<void>>{};
      v.reserve(cores-1);

      std::cout << "Adding " << N_add << " time-points at once" << std::endl;
      timer([&]() mutable {
	      for(std::remove_const_t<decltype(cores)> i = 0; i < cores-1; ++i){
		v.push_back(std::async(std::launch::async,
				       [&](){ f_core(b,N_add,n*i,n*(i+1)); }));
	      }
	      f_core(b,N_add,n*(cores-1),N_step);

	      for(auto& ve: v){ ve.wait(); }
	    },N_times);
    };

  std::cout << "Single-thread without lock" << std::endl;
  f(single,1);
  std::cout << std::endl;
  single.clear();
  f(single,100);
  std::cout << std::endl;

  std::cout << "Single-thread with lock" << std::endl;
  f(multi,1);
  std::cout << std::endl;
  multi.clear();
  f(multi,add_dim);
  std::cout << std::endl;

  std::cout << "Multi-thread with lock" << std::endl;
  const auto cores = std::thread::hardware_concurrency();
  std::cout << cores << " cores execution." << std::endl;

  multi_f(multi2,1,cores);
  std::cout << std::endl;
  multi2.clear();
  multi_f(multi2,add_dim,cores);
  std::cout << std::endl;

}

void test_MultiThreadPrioritizedSampler(){
  constexpr const auto buffer_size = 1024ul * 256ul;
  constexpr const auto alpha = 0.5;
  constexpr const auto beta = 0.4;
  constexpr const auto batch_size = 16ul;

  auto per = ymd::CppThreadSafePrioritizedSampler<Priority>(buffer_size,alpha);
}

int main(){

  constexpr const auto obs_dim = 3ul;
  constexpr const auto act_dim = 1ul;

  constexpr const auto N_buffer_size = 1024ul;
  constexpr const auto N_step = 3 * N_buffer_size;
  constexpr const auto N_batch_size = 16ul;

  constexpr const auto N_times = 1000ul;

  auto alpha = 0.7;
  auto beta = 0.5;

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

  auto rb = ymd::CppReplayBuffer<Observation,Action,Reward,Done>{N_buffer_size,
								 obs_dim,
								 act_dim};

  auto per = ymd::CppPrioritizedReplayBuffer<Observation,Action,
					     Reward,Done,Priority>{N_buffer_size,
								   obs_dim,
								   act_dim,
								   alpha};


  for(auto i = 0ul; i < N_step; ++i){
    auto obs = std::vector<Observation>(obs_dim,0.1*i);
    auto act = std::vector<Action>(act_dim,2.0*i);
    auto rew = 0.1 * i;
    auto next_obs = std::vector<Observation>(obs_dim,0.1*(i+1));
    auto done = (N_step - 1 == i) ? 1.0: 0.0;

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
    auto done = (N_step - 1 == i) ? 1.0: 0.0;

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


  std::cout << std::endl;
  std::cout << "PrioritizedSampler" << std::endl;
  auto ps = ymd::CppPrioritizedSampler(N_buffer_size,0.7);
  for(auto i = 0ul; i < N_step; ++i){
    ps.set_priorities(i % N_buffer_size,0.5);
  }

  auto ps_w = std::vector<Priority>{};
  auto ps_i = std::vector<std::size_t>{};

  ps.sample(N_batch_size,0.4,ps_w,ps_i,N_buffer_size);

  show_vector(ps_w,"weights [0.5,...,0.5]");
  show_vector(ps_i,"indexes [0.5,...,0.5]");

  ps_w[0] = 1e+10;
  ps.update_priorities(ps_i,ps_w);
  ps.sample(N_batch_size,0.4,ps_w,ps_i,N_buffer_size);
  show_vector(ps_w,"weights [0.5,.,1e+10,..,0.5]");
  show_vector(ps_i,"indexes [0.5,.,1e+10,..,0.5]");

  test_NstepReward();
  test_SelectiveEnvironment();

  test_MultiThreadRingEnvironment();

  test_MultiThreadPrioritizedSampler();

  return 0;
}
