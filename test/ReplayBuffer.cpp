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

#include "unittest.hh"

using namespace std::literals;

using Observation = double;
using Action = double;
using Reward = double;
using Done = double;
using Priority = double;

const auto cores = std::thread::hardware_concurrency();
using cores_t = std::remove_const_t<decltype(cores)>;

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
  ymd::show_pointer(obs_,se.get_stored_size()*obs_dim,"obs");
  ymd::show_pointer(act_,se.get_stored_size()*act_dim,"act");
  ymd::show_pointer(rew_,se.get_stored_size(),"rew");
  ymd::show_pointer(next_obs_,se.get_stored_size()*obs_dim,"next_obs");
  ymd::show_pointer(done_,se.get_stored_size(),"done");

  assert(1ul == ep_len);
  assert(1ul == se.get_next_index());
  assert(1ul == se.get_stored_size());
  assert(1ul == se.get_stored_episode_size());

  // Add remained 3-steps
  se.store(obs.data()+1,act.data()+1,rew.data()+1,obs.data()+2,done.data()+1,
	   episode_len - 1ul);
  se.get_episode(0,ep_len,obs_,act_,rew_,next_obs_,done_);
  ymd::show_pointer(obs_,se.get_stored_size()*obs_dim,"obs");
  ymd::show_pointer(act_,se.get_stored_size()*act_dim,"act");
  ymd::show_pointer(rew_,se.get_stored_size(),"rew");
  ymd::show_pointer(next_obs_,se.get_stored_size()*obs_dim,"next_obs");
  ymd::show_pointer(done_,se.get_stored_size(),"done");

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
  ymd::show_pointer(obs_,se.get_stored_size()*obs_dim,"obs");
  ymd::show_pointer(act_,se.get_stored_size()*act_dim,"act");
  ymd::show_pointer(rew_,se.get_stored_size(),"rew");
  ymd::show_pointer(next_obs_,se.get_stored_size()*obs_dim,"next_obs");
  ymd::show_pointer(done_,se.get_stored_size(),"done");

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
  ymd::show_pointer(obs_,se.get_stored_size()*obs_dim,"obs");
  ymd::show_pointer(act_,se.get_stored_size()*act_dim,"act");
  ymd::show_pointer(rew_,se.get_stored_size(),"rew");
  ymd::show_pointer(next_obs_,se.get_stored_size()*obs_dim,"next_obs");
  ymd::show_pointer(done_,se.get_stored_size(),"done");
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

void test_DimensionalBuffer(){
  constexpr const auto obs_dim = 3ul;

  constexpr const auto N_buffer_size = 1024ul;
  constexpr const auto N_batch_size = 16ul;

  constexpr const auto N_times = 1000ul;

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
  assert(0ul == obs_ptr[0]);
  assert(1ul == obs_ptr[1]);
  assert(2ul == obs_ptr[2]);

  for(auto n = 0ul; n < N_times; ++n){
    auto next_index = std::min(n*obs_dim % N_buffer_size,N_buffer_size-1);
    dm.store_data(v.data(),0ul,next_index,1ul);
    assert(0ul == obs_ptr[next_index + 0]);
    assert(1ul == obs_ptr[next_index + 1]);
    assert(2ul == obs_ptr[next_index + 2]);
  }
}

int main(){

  constexpr const auto obs_dim = 3ul;
  constexpr const auto act_dim = 1ul;
  constexpr const auto rew_dim = 1ul;

  constexpr const auto N_buffer_size = 1024ul;
  constexpr const auto N_batch_size = 16ul;

  constexpr const auto N_times = 1000ul;

  auto alpha = 0.7;
  auto beta = 0.5;


  std::cout << std::endl;
  std::cout << "PrioritizedSampler" << std::endl;
  auto ps = ymd::CppPrioritizedSampler(N_buffer_size,0.7);
  for(auto i = 0ul; i < N_step; ++i){
    ps.set_priorities(i % N_buffer_size,0.5);
  }

  auto ps_w = std::vector<Priority>{};
  auto ps_i = std::vector<std::size_t>{};

  ps.sample(N_batch_size,0.4,ps_w,ps_i,N_buffer_size);

  ymd::show_vector(ps_w,"weights [0.5,...,0.5]");
  ymd::show_vector(ps_i,"indexes [0.5,...,0.5]");

  ps_w[0] = 1e+10;
  ps.update_priorities(ps_i,ps_w);
  ps.sample(N_batch_size,0.4,ps_w,ps_i,N_buffer_size);
  ymd::show_vector(ps_w,"weights [0.5,.,1e+10,..,0.5]");
  ymd::show_vector(ps_i,"indexes [0.5,.,1e+10,..,0.5]");

  test_SelectiveEnvironment();

  return 0;
}
