// Copyright (c) 2009-2021, Tor M. Aamodt, Vijay Kandiah, Nikos Hardavellas,
// Mahmoud Khairy, Junrui Pan, Timothy G. Rogers
// The University of British Columbia, Northwestern University, Purdue
// University All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
// 1. Redistributions of source code must retain the above copyright notice,
// this
//    list of conditions and the following disclaimer;
// 2. Redistributions in binary form must reproduce the above copyright notice,
//    this list of conditions and the following disclaimer in the documentation
//    and/or other materials provided with the distribution;
// 3. Neither the names of The University of British Columbia, Northwestern
//    University nor the names of their contributors may be used to
//    endorse or promote products derived from this software without specific
//    prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
// SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
// INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
// CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
// ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.

#ifndef MC_PARTITION_INCLUDED
#define MC_PARTITION_INCLUDED

#include "../abstract_hardware_model.h"
#include "dram.h"
#include "mc_cache.h"
#include "oracle.h"

#include <list>
#include <queue>
#include <unordered_map>

class mem_fetch;

class partition_mf_allocator : public mem_fetch_allocator {
 public:
  partition_mf_allocator(const memory_config *config) {
    m_memory_config = config;
  }
  virtual mem_fetch *alloc(const class warp_inst_t &inst,
                           const mem_access_t &access,
                           unsigned long long cycle) const {
    abort();
    return NULL;
  }
  virtual mem_fetch *alloc(new_addr_type addr, mem_access_type type,
                           unsigned size, bool wr, unsigned long long cycle,
                           unsigned long long streamID) const;
  virtual mem_fetch *alloc(new_addr_type addr, mem_access_type type,
                           const active_mask_t &active_mask,
                           const mem_access_byte_mask_t &byte_mask,
                           const mem_access_sector_mask_t &sector_mask,
                           unsigned size, bool wr, unsigned long long cycle,
                           unsigned wid, unsigned sid, unsigned tpc,
                           mem_fetch *original_mf,
                           unsigned long long streamID) const;

 private:
  const memory_config *m_memory_config;
};

// Memory partition unit contains all the units assolcated with a single DRAM
// channel.
// - It arbitrates the DRAM channel among multiple sub partitions.
// - It does not connect directly with the interconnection network.
class memory_partition_unit {
 public:
  memory_partition_unit(unsigned partition_id, const memory_config *config,
                        class memory_stats_t *stats, class gpgpu_sim *gpu);
  ~memory_partition_unit();

  bool busy() const;

  void cache_cycle(unsigned cycle);
  void dram_cycle();
  void simple_dram_model_cycle();

  void set_done(mem_fetch *mf);

  void visualizer_print(gzFile visualizer_file) const;
  void print_stat(FILE *fp) { m_dram->print_stat(fp); }
  void visualize() const { m_dram->visualize(); }
  void print(FILE *fp) const;
  void handle_memcpy_to_gpu(size_t dst_start_addr, unsigned subpart_id,
                            mem_access_sector_mask_t mask);

  class memory_sub_partition *get_sub_partition(int sub_partition_id) {
    return m_sub_partition[sub_partition_id];
  }

  // Power model
  void set_dram_power_stats(unsigned &n_cmd, unsigned &n_activity,
                            unsigned &n_nop, unsigned &n_act, unsigned &n_pre,
                            unsigned &n_rd, unsigned &n_wr, unsigned &n_wr_WB,
                            unsigned &n_req) const;

  int global_sub_partition_id_to_local_id(int global_sub_partition_id) const;

  unsigned get_mpid() const { return m_id; }

  class gpgpu_sim *get_mgpu() const { return m_gpu; }

  fifo_pipeline<mem_fetch>* m_prefetch_global_queue = nullptr;
  mem_fetch* generate_prefetch_after_issue();
  mem_fetch* new_prefetch_req(new_addr_type addr, mem_fetch* original);

  enum prefetch_state_t { PF_PENDING = 0, PF_ARRIVED = 1 };

  //std::unordered_map<new_addr_type, prefetch_state_t> m_prefetch_table;

  std::list<new_addr_type> m_pf_lru;

  struct PrefetchEntry {
    prefetch_state_t state;
    unsigned long long ts;
    std::list<new_addr_type>::iterator it;
  };
  std::unordered_map<new_addr_type, PrefetchEntry> m_prefetch_table;

  size_t m_pf_capacity = 512;

  unsigned long long now();

  inline void pf_cleanup_expired() {
    if (m_prefetch_table.size() < m_pf_capacity)
        return;

    unsigned long long cur = now();
    auto oldest = m_prefetch_table.end();
    unsigned long long max_age = 2000;

    for (auto tit = m_prefetch_table.begin(); tit != m_prefetch_table.end(); ++tit) {
        unsigned long long age = cur - tit->second.ts;
        if (age > max_age) {
            max_age = age;
            oldest = tit;
        }
    }

    if (oldest != m_prefetch_table.end()) {
        m_pf_lru.erase(oldest->second.it);
        m_prefetch_table.erase(oldest);
    }
  }

  inline void pf_track_request(new_addr_type addr) {
  auto it = m_prefetch_table.find(addr);
  if (it != m_prefetch_table.end()) {
    m_pf_lru.erase(it->second.it);
    m_pf_lru.push_front(addr);
    it->second.it = m_pf_lru.begin();
    it->second.ts = now();
    it->second.state = PF_PENDING; 
    return;
  }

  if (m_prefetch_table.size() >= m_pf_capacity && !m_pf_lru.empty()) {
    new_addr_type old = m_pf_lru.back();
    m_pf_lru.pop_back();
    m_prefetch_table.erase(old);
  }

  m_pf_lru.push_front(addr);
  PrefetchEntry e;
  e.state = PF_PENDING;
  e.ts = now();
  e.it = m_pf_lru.begin();
  m_prefetch_table.emplace(addr, e);
  }

inline void pf_mark_arrived(new_addr_type addr) {
  auto it = m_prefetch_table.find(addr);
  if (it == m_prefetch_table.end()) {
    if (m_prefetch_table.size() >= m_pf_capacity && !m_pf_lru.empty()) {
      new_addr_type old = m_pf_lru.back();
      m_pf_lru.pop_back();
      m_prefetch_table.erase(old);
    }
    m_pf_lru.push_front(addr);
    PrefetchEntry e;
    e.state = PF_ARRIVED;
    e.ts = now();
    e.it = m_pf_lru.begin();
    m_prefetch_table.emplace(addr, e);
    return;
  }
  
  m_pf_lru.erase(it->second.it);
  m_pf_lru.push_front(addr);
  it->second.it = m_pf_lru.begin();
  it->second.ts = now();
  it->second.state = PF_ARRIVED;
}

  inline bool pf_exists(new_addr_type addr) const {
    auto it = m_prefetch_table.find(addr);
    return it != m_prefetch_table.end();
  }

  inline bool pf_is_arrived(new_addr_type addr) const {
    auto it = m_prefetch_table.find(addr);
    return (it != m_prefetch_table.end()) && (it->second.state == PF_ARRIVED);
  }

  struct sram_delay_t { unsigned long long ready_cycle; mem_fetch* req; };
  std::list<sram_delay_t> m_sram_ready;

  std::unordered_map<new_addr_type, mem_fetch*> m_sram_unready;

  std::vector<unsigned> m_bank_inflight;
  std::vector<std::unordered_map<unsigned /*row*/, unsigned /*cnt*/>> m_bank_row_pending;

  int bank_id_from_mf(class mem_fetch* mf);

  OraclePrefetcher m_oracle;

  static const unsigned RLB_SIZE = 256;

  struct rlb_entry_t {
    new_addr_type line_addr; 
    bool valid;
  };

  rlb_entry_t m_recent_lines[RLB_SIZE];
  unsigned m_rlb_head;

  void rlb_insert(new_addr_type line);

  bool rlb_contains(new_addr_type line);

  new_addr_type pick_prefetch_addr_from_pattern();

  static const unsigned ACTIVE_BASE_TABLE_SIZE = 32;

  struct active_base_entry_t {
    new_addr_type base;            
    new_addr_type last_addr;        
    bool valid;
    unsigned long long timestamp;    
  };

  active_base_entry_t m_active_base_table[ACTIVE_BASE_TABLE_SIZE];

  new_addr_type align_active_base(new_addr_type addr);

  int find_active_base_slot(new_addr_type base);

  void update_active_base_table(new_addr_type addr);

    struct PrefetchMemEntry {
    //unsigned long long time;
    uint64_t base;
    uint64_t addr;
    uint64_t appear;   // appear time of addr (3rd column of mpool.txt)
    //unsigned long long num;
  };

  std::vector<PrefetchMemEntry> g_prefetch_mem_table;

 struct PoolCand {
    new_addr_type addr;
    int bank;
    unsigned row;
    unsigned time;
    unsigned long long appear;   // appear time carried from mpool.txt
  };

  new_addr_type STEP_SMALL = 0x20;
  new_addr_type STEP_BIG   = 0x2000;

  std::vector<PoolCand> mpool;

  mem_fetch* m_prefetch_template = nullptr;
  void make_prefetch_from_template(mem_fetch* original);

  void active_mpool(); 

  std::vector<unsigned long long> mrecord;

  void broadcast(unsigned long long addr) {
    mrecord.push_back(addr);
  }

  unsigned sram_max = 0;

 private:
  unsigned m_id;
  const memory_config *m_config;
  class memory_stats_t *m_stats;
  class memory_sub_partition **m_sub_partition;
  class dram_t *m_dram;

  class arbitration_metadata {
   public:
    arbitration_metadata(const memory_config *config);

    // check if a subpartition still has credit
    bool has_credits(int inner_sub_partition_id) const;
    // borrow a credit for a subpartition
    void borrow_credit(int inner_sub_partition_id);
    // return a credit from a subpartition
    void return_credit(int inner_sub_partition_id);

    // return the last subpartition that borrowed credit
    int last_borrower() const { return m_last_borrower; }

    void print(FILE *fp) const;

   private:
    // id of the last subpartition that borrowed credit
    int m_last_borrower;

    int m_shared_credit_limit;
    int m_private_credit_limit;

    // credits borrowed by the subpartitions
    std::vector<int> m_private_credit;
    int m_shared_credit;
  };
  arbitration_metadata m_arbitration_metadata;

  // determine wheither a given subpartition can issue to DRAM
  bool can_issue_to_dram(int inner_sub_partition_id);

  // model DRAM access scheduler latency (fixed latency between L2 and DRAM)
  struct dram_delay_t {
    unsigned long long ready_cycle;
    class mem_fetch *req;
  };
  std::list<dram_delay_t> m_dram_latency_queue;

  class gpgpu_sim *m_gpu;
};

class memory_sub_partition {
 public:
  memory_sub_partition(unsigned sub_partition_id, const memory_config *config,
                       class memory_stats_t *stats, class gpgpu_sim *gpu);
  ~memory_sub_partition();

  unsigned get_id() const { return m_id; }

  bool busy() const;

  void cache_cycle(unsigned cycle);

  bool full() const;
  bool full(unsigned size) const;
  void push(class mem_fetch *mf, unsigned long long clock_cycle);
  class mem_fetch *pop();
  class mem_fetch *top();
  void set_done(mem_fetch *mf);

  unsigned flushL2();
  unsigned invalidateL2();

  // interface to L2_dram_queue
  bool L2_dram_queue_empty() const;
  class mem_fetch *L2_dram_queue_top() const;
  void L2_dram_queue_pop();

  // interface to dram_L2_queue
  bool dram_L2_queue_full() const;
  void dram_L2_queue_push(class mem_fetch *mf);

  void visualizer_print(gzFile visualizer_file);
  void print_cache_stat(unsigned &accesses, unsigned &misses) const;
  void print(FILE *fp) const;

  void accumulate_L2cache_stats(class cache_stats &l2_stats) const;
  void get_L2cache_sub_stats(struct cache_sub_stats &css) const;

  // Support for getting per-window L2 stats for AerialVision
  void get_L2cache_sub_stats_pw(struct cache_sub_stats_pw &css) const;
  void clear_L2cache_stats_pw();

  void force_l2_tag_update(new_addr_type addr, unsigned time,
                           mem_access_sector_mask_t mask) {
    m_L2cache->force_tag_access(addr, m_memcpy_cycle_offset + time, mask);
    m_memcpy_cycle_offset += 1;
  }

 private:
  // data
  unsigned m_id;  //< the global sub partition ID
  const memory_config *m_config;
  class l2_cache *m_L2cache;
  class L2interface *m_L2interface;
  class gpgpu_sim *m_gpu;
  partition_mf_allocator *m_mf_allocator;

  // model delay of ROP units with a fixed latency
  struct rop_delay_t {
    unsigned long long ready_cycle;
    class mem_fetch *req;
  };
  std::queue<rop_delay_t> m_rop;

  // these are various FIFOs between units within a memory partition
  fifo_pipeline<mem_fetch> *m_icnt_L2_queue;
  fifo_pipeline<mem_fetch> *m_L2_dram_queue;
  fifo_pipeline<mem_fetch> *m_dram_L2_queue;
  fifo_pipeline<mem_fetch> *m_L2_icnt_queue;  // L2 cache hit response queue

  class mem_fetch *L2dramout;
  unsigned long long int wb_addr;

  class memory_stats_t *m_stats;

  std::set<mem_fetch *> m_request_tracker;

  friend class L2interface;

  std::vector<mem_fetch *> breakdown_request_to_sector_requests(mem_fetch *mf);

  // This is a cycle offset that has to be applied to the l2 accesses to account
  // for the cudamemcpy read/writes. We want GPGPU-Sim to only count cycles for
  // kernel execution but we want cudamemcpy to go through the L2. Everytime an
  // access is made from cudamemcpy this counter is incremented, and when the l2
  // is accessed (in both cudamemcpyies and otherwise) this value is added to
  // the gpgpu-sim cycle counters.
  unsigned m_memcpy_cycle_offset;
};

class L2interface : public mem_fetch_interface {
 public:
  L2interface(memory_sub_partition *unit) { m_unit = unit; }
  virtual ~L2interface() {}
  virtual bool full(unsigned size, bool write) const {
    // assume read and write packets all same size
    return m_unit->m_L2_dram_queue->full();
  }
  virtual void push(mem_fetch *mf) {
    mf->set_status(IN_PARTITION_L2_TO_DRAM_QUEUE, 0 /*FIXME*/);
    m_unit->m_L2_dram_queue->push(mf);
  }

 private:
  memory_sub_partition *m_unit;
};

#endif
