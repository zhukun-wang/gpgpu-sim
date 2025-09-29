// mc_sram_cache.h
#pragma once

#include <vector>
#include <cstddef>
#include <cassert>

#include "../abstract_hardware_model.h"  // for new_addr_type

enum pf_state { PF_MISS = 0, PF_INFLIGHT = 1, PF_READY = 2 };

class mc_sram_cache {
 public:
  mc_sram_cache(size_t size_bytes, unsigned line_sz, unsigned assoc,
                unsigned hit_lat)
      : m_line(line_sz),
        m_assoc(assoc),
        m_hit_lat(hit_lat) {
    assert(m_line > 0);
    assert((m_line & (m_line - 1)) == 0 && "line_sz must be power of two");
    assert(m_assoc > 0);
    assert(size_bytes % (size_bytes ? (m_line * m_assoc) : 1) == 0);
    m_sets = (size_bytes / m_line) / m_assoc;
    assert(m_sets > 0 && "size_bytes too small for given line_sz/assoc");

    m_table.assign(m_sets, std::vector<sram_entry>(m_assoc));
    m_rr.assign(m_sets, 0);  
  }

  inline pf_state probe(new_addr_type addr) const {
    const new_addr_type la = line_addr(addr);
    const size_t idx = set_index(la);
    const auto &set = m_table[idx];
    for (const auto &e : set) {
      if (e.valid && e.tag == la) return e.hit ? PF_READY : PF_INFLIGHT;
    }
    return PF_MISS;
  }

  inline void allocate_inflight(new_addr_type addr) {
    const new_addr_type la = line_addr(addr);
    const size_t idx = set_index(la);
    auto &set = m_table[idx];

    for (auto &e : set) {
      if (e.valid && e.tag == la) return;
    }
    
    size_t way = find_invalid_way(set);
    if (way == m_assoc) {
      way = m_rr[idx];
      m_rr[idx] = (m_rr[idx] + 1) % m_assoc;
    }
    set[way].tag = la;
    set[way].hit = false;
    set[way].valid = true;
  }

  inline void mark_ready(new_addr_type addr) {
    const new_addr_type la = line_addr(addr);
    const size_t idx = set_index(la);
    auto &set = m_table[idx];

    for (auto &e : set) {
      if (e.valid && e.tag == la) { e.hit = true; return; }
    }
    size_t way = find_invalid_way(set);
    if (way == m_assoc) {
      way = m_rr[idx];
      m_rr[idx] = (m_rr[idx] + 1) % m_assoc;
    }
    set[way].tag = la;
    set[way].hit = true;
    set[way].valid = true;
  }

  inline void invalidate(new_addr_type addr) {
    const new_addr_type la = line_addr(addr);
    auto &set = m_table[set_index(la)];
    for (auto &e : set) {
      if (e.valid && e.tag == la) { e.valid = false; e.hit = false; return; }
    }
  }

  inline void clear() {
    for (auto &set : m_table) {
      for (auto &e : set) { e.valid = false; e.hit = false; e.tag = 0; }
    }
    std::fill(m_rr.begin(), m_rr.end(), 0);
  }

  inline unsigned hit_latency() const { return m_hit_lat; }

  inline unsigned line_size() const { return m_line; }
  inline unsigned assoc() const { return m_assoc; }
  inline size_t   num_sets() const { return m_sets; }
  inline size_t   capacity_bytes() const { return m_sets * m_assoc * size_t(m_line); }

 private:
  struct sram_entry {
    new_addr_type tag = 0;
    bool hit = false;   
    bool valid = false;  
  };

  inline new_addr_type line_addr(new_addr_type a) const {
    return a & ~(new_addr_type(m_line) - 1ULL);
  }
  inline size_t set_index(new_addr_type line_addr) const {
    return size_t((line_addr / m_line) % m_sets);
  }
  inline size_t find_invalid_way(const std::vector<sram_entry> &set) const {
    for (size_t i = 0; i < m_assoc; ++i) if (!set[i].valid) return i;
    return m_assoc; 
  }

 private:
    unsigned m_line;   
    unsigned m_assoc;  
    unsigned m_hit_lat; 
    size_t   m_sets;   

    std::vector<std::vector<sram_entry>> m_table; 
    std::vector<size_t> m_rr;                   
};

