#pragma once
#include <cstdint>
#include <unordered_map>
#include <unordered_set>
#include <cstdio>

static constexpr uint32_t SECTOR32B_SHIFT = 5;   // 32B
static constexpr uint32_t LINE4KB_SHIFT   = 12;  // 4KB
static constexpr uint32_t TRUNK64KB_SHIFT = 16;  // 64KB
static constexpr uint32_t LINES_PER_CHUNK = 16;             // 64KB / 4KB
static constexpr uint16_t FULL_BITMAP     = (1u<<16) - 1;   // 16 bits

struct Ids {
  uint64_t chunk_id;  
  uint16_t sector_id; 
  uint8_t  line_id;   
  static inline Ids decode(uint64_t addr){
    Ids o;
    o.chunk_id  = (addr >> TRUNK64KB_SHIFT);
    o.sector_id = uint16_t((addr >> SECTOR32B_SHIFT) & 0x7F);
    o.line_id   = uint8_t((addr >> LINE4KB_SHIFT) & 0x0F);
    return o;
  }
};

struct Key {
  uint64_t chunk; uint16_t sector;
  bool operator==(const Key& b)const{ return chunk==b.chunk && sector==b.sector; }
};
struct KeyHash {
  size_t operator()(const Key& k) const {
    return std::hash<uint64_t>()((k.chunk<<16) ^ k.sector);
  }
};

class ChunkMonitor {
public:
  void enable(bool on){ enabled_ = on; }
  bool enabled() const { return enabled_; }

  inline void observe_read(uint64_t addr, unsigned long long abs_cycle){
    if (!enabled_) return;
    Ids ids = Ids::decode(addr);
    Key key{ids.chunk_id, ids.sector_id};
    if (completed_.find(key)!=completed_.end()) return;
    uint16_t &bm = bitmap_[key];
    uint16_t before = bm;
    bm |= (1u << ids.line_id);
    if (before!=bm && bm==FULL_BITMAP){
      //FILE *f = fopen("count.txt", "a");
      //fprintf(f, "Time: %u Chunk: %llu Sector: %u\n", abs_cycle, (unsigned long long)key.chunk, (unsigned)key.sector);
      //fclose(f);
      completed_.insert(key);
      // bitmap_.erase(key);
    }
  }

  void reset(){ bitmap_.clear(); completed_.clear(); }

private:
  bool enabled_ = false; 
  std::unordered_map<Key,uint16_t,KeyHash> bitmap_;
  std::unordered_set<Key,KeyHash>          completed_;
};

