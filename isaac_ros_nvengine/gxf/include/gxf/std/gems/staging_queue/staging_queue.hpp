/*
Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.

NVIDIA CORPORATION and its licensors retain all intellectual property
and proprietary rights in and to this software, related documentation
and any modifications thereto. Any use, reproduction, disclosure or
distribution of this software and related documentation without an express
license agreement from NVIDIA CORPORATION is strictly prohibited.
*/
#ifndef NVIDIA_GXF_STD_GEMS_STAGING_QUEUE_HPP
#define NVIDIA_GXF_STD_GEMS_STAGING_QUEUE_HPP

#include <mutex>
#include <utility>
#include <vector>

#include "core/assert.hpp"
#include "gxf/std/gems/staging_queue/staging_queue_iterator.hpp"

namespace gxf {
namespace staging_queue {

// Defines the behavior of a StagingQueue in case an element is added while the queue is already
// full.
enum class OverflowBehavior {
  // Pops oldest item to make room for the incoming item.
  kPop,
  // Rejects the incoming item.
  kReject,
  // The queue goes into an error state. Currently this raises a PANIC.
  kFault
};

// A thread-safe double-buffered queue implemented based on a ring buffer.
//
// This data structure provides a queue-like interface to a collection of items. When items are
// added to the queue with the function push they are placed into a backstage area. The pop
// function and query functions like size, peek or latest only operate on items on
// the main stage. Items are moved from the backstage to the main stage by calling the sync
// function. The queue only provides a fixed number of slots for elements. If elements are added to
// a full queue a user-selected strategy is used to determine how to proceed. Note that the overflow
// strategy can by triggerd when calling push in case the backstage area is full, or when calling
// sync in case the main stage is too full to receive all items from the backstage. When items are
// removed from the queue they are overwritten with a user-defined default value. This is useful
// for example in case shared pointers are stored in the queue. All functions of this type are
// protected by a common mutex thus making this a thread-safe type. The queue is implemented as a
// ring buffer internally.
//
// The following diagram depicts an example of a queue with a total capacity of four items. It
// currently stores three items in the main stage and two items in the backstage. O indicates
// empty slots, X indicates (filled) main stage slots and B indicates (filled) backstage slots.
// There are three empty slots as neither main stage nor backstage are at full capacity. If the
// sync function would be called in this situation the overflow behavior would be triggered as
// the main stage does not have enough room to receive all items from the backstage.
//
//     # start of ringbuffer           # end of ringbuffer
//     |                               |
//     | O | X | X | X | B | B | O | O |
//         |           |       |
//         |           |       # end
//         # begin     |
//                     # backstage
//
template <typename T>
class StagingQueue {
 public:
  using const_iterator_t = StagingQueueIterator<T>;

  // Creates a new staging queue. 'capacity' indicates the maximum number of elements allowed in
  // the queue. Note that the queue will allocate memory to hold two times capacity slots of item
  // type T. 'overflow_behevior' defines what will happen if a new item is added to a full queue.
  // 'null' is the default element which is used for empty slots. If an element is removed from the
  // queue the slot it occupied is set to this value.
  StagingQueue(size_t capacity, OverflowBehavior overflow_behavior, T null);

  // Creates an empty staging queue with no capacity and Fault overflow behavior
  StagingQueue();

  // Gets the overflow behavior which is used by this queue
  OverflowBehavior overflow_behavior() const;

  // Returns true if there are no elements in the main stage of the queue.
  // Identical to 'size() == 0;'
  bool empty() const;
  // Returns the number of elements in the main stage of the queue.
  size_t size() const;
  // Returns the maximum number of elements which can be stored in the main stage.
  size_t capacity() const;

  // Returns the number of elements in the back main stage of the queue.
  size_t back_size() const;

  // Gets the item at position 'index' in the main stage of the queue starting with the oldest item.
  // Returns a reference to the null object if there are no items in the main stage.
  const T& peek(size_t index = 0) const;
  // Gets the item at position 'index' in the back stage of the queue starting with the oldest item.
  // Returns a reference to the null object if there are no items in the back stage.
  const T& peek_backstage(size_t index = 0) const;
  // Gets the item at position 'index' in the main stage of the queue starting with the newest item.
  // Returns a reference to the null object if there are no items in the main stage.
  // Identical to 'peek(size() - 1 - index)'.
  const T& latest(size_t index = 0) const;

  // Gives an iterator pointing to the first element in the main stage.
  const_iterator_t begin() const;
  // Gives an iterator pointing to the element after the last element in the main stage.
  const_iterator_t end() const;

  // Removes the oldest item from the queue's main stage and returns it. In case there are no items
  // in the main stage this function simply returns a copy of the null object. This function
  // invalidates iterators returned by begin() or end().
  T pop();
  // Removes all items from the main stage. This function invalidates iterators in the same way
  // as the pop() function.
  void popAll();

  // Adds a new item to the back stage. In case the back stage is at capacity the overflow behavior
  // is used to decide what to do. The added item will NOT be visible to any other functions which
  // operate on the main stage, like size, peek, pop or similar, until 'sync' is called.
  bool push(T item);

  // Moves all items from the back stage to the main stage. In case the main stage is too full
  // to receive all items from the back stage the specified overflow behavior is used to decide what
  // to do. This function invalidates iterators in the same way as the pop() function.
  bool sync();

 private:
  // Returns the item in the underlying ringbuffer which is at position 'index' relative to the
  // start of the ring buffer.
  const T& at(size_t index) const { return items_[index % items_.size()]; }
  T& at(size_t index) { return items_[index % items_.size()]; }

  // The maximum number of items in the main stage and back stage. Total number items in the queue
  // is two times capacity.
  const size_t capacity_;
  // This behavior defines if either push or sync are called but the main stage contains too many
  // items already.
  const OverflowBehavior overflow_behavior_;

  // The null value is used for empty slots. It is also used as return value for peek and latest
  // in case the given index is out of bounds.
  T null_;

  // The container which holds the elements of the main stage and back stage. The container
  // allocates memory in the constructor and does not change it's capacity thereafter.
  std::vector<T> items_;

  // Index of the first element of the main stage. The value of begin_ is guaranteed to be in the
  // range {0, ..., items_.size() - 1}.
  size_t begin_;

  // Number of elements in the mainstage.
  size_t num_mainstage_;
  // Number of elements in the backstage.
  size_t num_backstage_;

  // This mutext protects all functions from concurrent access.
  mutable std::mutex mutex_;
};

//--------------------------------------------------------------------------------------------------

template <typename T>
StagingQueue<T>::StagingQueue(size_t capacity, OverflowBehavior overflow_behavior, T null)
    : capacity_(capacity),
      overflow_behavior_(overflow_behavior),
      null_(null),
      items_(2 * capacity, null),
      begin_(0),
      num_mainstage_(0),
      num_backstage_(0) {}

template <typename T>
StagingQueue<T>::StagingQueue()
    : capacity_(0),
      overflow_behavior_(OverflowBehavior::kFault),
      null_(),
      items_(0),
      begin_(0),
      num_mainstage_(0),
      num_backstage_(0) {}

template <typename T>
OverflowBehavior StagingQueue<T>::overflow_behavior() const {
  // Mutex not necessary as overflow_behavior_ is immutable.
  return overflow_behavior_;
}

template <typename T>
bool StagingQueue<T>::empty() const {
  std::lock_guard<std::mutex> lock(mutex_);
  return num_mainstage_ == 0;
}

template <typename T>
size_t StagingQueue<T>::size() const {
  std::lock_guard<std::mutex> lock(mutex_);
  return num_mainstage_;
}

template <typename T>
size_t StagingQueue<T>::capacity() const {
  // Mutex not necessary as capacity_ is immutable.
  return capacity_;
}

template <typename T>
size_t StagingQueue<T>::back_size() const {
  std::lock_guard<std::mutex> lock(mutex_);
  return num_backstage_;
}

template <typename T>
const T& StagingQueue<T>::peek(size_t index) const {
  std::lock_guard<std::mutex> lock(mutex_);
  if (index < num_mainstage_) {
    return at(begin_ + index);
  } else {
    return null_;
  }
}

template <typename T>
const T& StagingQueue<T>::peek_backstage(size_t index) const {
  std::lock_guard<std::mutex> lock(mutex_);
  if (index < num_backstage_) {
    const size_t begin_backstage = begin_ + num_mainstage_;
    return at(begin_backstage + index);
  } else {
    return null_;
  }
}

template <typename T>
const T& StagingQueue<T>::latest(size_t index) const {
  std::lock_guard<std::mutex> lock(mutex_);
  if (index < num_mainstage_) {
    return at(begin_ + num_mainstage_ - index - 1);
  } else {
    return null_;
  }
}

template <typename T>
typename StagingQueue<T>::const_iterator_t StagingQueue<T>::begin() const {
  return const_iterator_t(items_.data(), items_.size(), begin_);
}

template <typename T>
typename StagingQueue<T>::const_iterator_t StagingQueue<T>::end() const {
  return const_iterator_t(items_.data(), items_.size(), begin_ + num_mainstage_);
}

template <typename T>
T StagingQueue<T>::pop() {
  std::lock_guard<std::mutex> lock(mutex_);
  T result = null_;
  if (num_mainstage_ > 0) {
    std::swap(at(begin_++), result);
    num_mainstage_--;
  }
  begin_ %= items_.size();
  return result;
}

template <typename T>
void StagingQueue<T>::popAll() {
  std::lock_guard<std::mutex> lock(mutex_);
  const size_t begin_backstage = begin_ + num_mainstage_;
  while (begin_ < begin_backstage) {
    at(begin_++) = null_;
  }
  begin_ %= items_.size();
  num_mainstage_ = 0;
}

template <typename T>
bool StagingQueue<T>::push(T item) {
  std::lock_guard<std::mutex> lock(mutex_);
  const size_t begin_backstage = begin_ + num_mainstage_;
  if (num_backstage_ == capacity_) {
    // Trying to add a item to a full backstage.
    switch (overflow_behavior_) {
      case OverflowBehavior::kPop: {
        // Move items in back stage one over. This removes the oldest item and makes room for one
        // new item.
        const size_t backstage_end = begin_backstage + num_backstage_;
        for (size_t i = begin_backstage + 1; i < backstage_end; i++) {
          at(i - 1) = std::move(at(i));
        }
        // Add new item at the end of the backstage.
        at(backstage_end - 1) = std::move(item);
      } break;
      case OverflowBehavior::kReject:
        // Don't add the new item to the backstage.
        break;
      case OverflowBehavior::kFault:
        return false;
        // FIXME "Added an item to a full queue while using the 'Fault' overflow behavior."
      default:
        // FIXME "Invalid parameter"
        return false;
    }
  } else {
    // Add the new item to the end of the backstage.
    at(begin_backstage + num_backstage_) = std::move(item);
    num_backstage_++;
  }
  return true;
}

template <typename T>
bool StagingQueue<T>::sync() {
  std::lock_guard<std::mutex> lock(mutex_);
  // Add back stage items to main stage
  num_mainstage_ += num_backstage_;
  num_backstage_ = 0;
  // Handle overflow if necessary
  if (num_mainstage_ > capacity_) {
    switch (overflow_behavior_) {
      case OverflowBehavior::kPop: {
        // Make sure that main and back stage together don't execeed capacity. Remove excess items
        // starting with the first item in the main stage. This effectively pops the oldest items to
        // make room for new items.
        const size_t new_begin = begin_ + num_mainstage_ - capacity_;
        while (begin_ < new_begin) {
          at(begin_++) = null_;
        }
        num_mainstage_ = capacity_;
      } break;
      case OverflowBehavior::kReject:
        // Make sure that main and back stage together don't execeed capacity. Remove excess items
        // starting with the last item in the back stage. This effectively rejects new items.
        while (num_mainstage_ > capacity_) {
          at(begin_ + --num_mainstage_) = null_;
        }
        break;
      case OverflowBehavior::kFault:
        return false;
        // FIXME "Added items to a full queue while using the 'Fault' overflow behavior.";
      default:
        // FIXME "Invalid parameter"
        return false;
    }
  }
  begin_ %= items_.size();
  return true;
}

}  // namespace staging_queue
}  // namespace gxf

#endif
