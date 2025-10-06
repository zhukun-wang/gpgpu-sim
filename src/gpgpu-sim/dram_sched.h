// Copyright (c) 2009-2011, Tor M. Aamodt, Ali Bakhoda, George L. Yuan
// The University of British Columbia
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
// Redistributions of source code must retain the above copyright notice, this
// list of conditions and the following disclaimer.
// Redistributions in binary form must reproduce the above copyright notice,
// this list of conditions and the following disclaimer in the documentation
// and/or other materials provided with the distribution. Neither the name of
// The University of British Columbia nor the names of its contributors may be
// used to endorse or promote products derived from this software without
// specific prior written permission.
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

#ifndef dram_sched_h_INCLUDED
#define dram_sched_h_INCLUDED

#include <list>
#include <map>
#include <unordered_map>
#include <cstdint>
#include "dram.h"
#include "gpu-misc.h"
#include "gpu-sim.h"
#include "shader.h"

#include "oracle.h"

enum memory_mode { READ_MODE = 0, WRITE_MODE };

class frfcfs_scheduler {
 public:
  frfcfs_scheduler(const memory_config *config, dram_t *dm,
                   memory_stats_t *stats);
  void add_req(dram_req_t *req);
  void data_collection(unsigned bank);
  dram_req_t *schedule(unsigned bank, unsigned curr_row);
  void print(FILE *fp);
  unsigned num_pending() const { return m_num_pending; }
  unsigned num_write_pending() const { return m_num_write_pending; }

  unsigned last_detect_time;
  unsigned seq128_num;
  unsigned total_req_num;
  bool chunk_sig;
  
  struct stream_entry_t {
    bool     valid = false;
    unsigned row = 0;
    unsigned last_col = 0;
    int      stride = 0;
    unsigned conf = 0;
    unsigned len_cl = 0;
    uint64_t last_ts = 0;
    class mem_fetch *last_data;
    class gpgpu_sim *last_m_gpu;
  };

  int find_match_stream(const std::vector<stream_entry_t>& tbl, unsigned row, unsigned col, uint64_t now);
  int alloc_stream(std::vector<stream_entry_t>& tbl);
  void start_stream(stream_entry_t& e, unsigned row, unsigned col, uint64_t now);
  int update_stream(unsigned bank_id, stream_entry_t& e, unsigned col, uint64_t now);
  void finalize_stream_if_chunk(stream_entry_t& e);


 private:
  const memory_config *m_config;
  dram_t *m_dram;
  unsigned m_num_pending;
  unsigned m_num_write_pending;
  std::list<dram_req_t *> *m_queue;
  std::map<unsigned, std::list<std::list<dram_req_t *>::iterator> > *m_bins;
  std::list<std::list<dram_req_t *>::iterator> **m_last_row;
  unsigned *curr_row_service_time;  // one set of variables for each bank.
  unsigned *row_service_timestamp;  // tracks when scheduler began servicing
                                    // current row

  std::list<dram_req_t *> *m_write_queue;
  std::map<unsigned, std::list<std::list<dram_req_t *>::iterator> >
      *m_write_bins;
  std::list<std::list<dram_req_t *>::iterator> **m_last_write_row;

  enum memory_mode m_mode;
  memory_stats_t *m_stats;

  std::vector<std::vector<stream_entry_t>> m_stream_tbl;
  
  static constexpr int      MAX_STREAMS_PER_BANK = 8;
  static constexpr unsigned MIN_CONF             = 3;
  static constexpr unsigned MIN_CHUNK_CL         = 4;
  static constexpr unsigned MAX_COL_GAP          = 1;
  static constexpr uint64_t STREAM_TIMEOUT       = 200;
  static constexpr unsigned PREFETCH_CONF_TH     = 8;

  OraclePrefetcher m_oracle;
};

#endif
