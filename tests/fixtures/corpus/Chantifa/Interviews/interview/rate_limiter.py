## Complete Solution
import time

class RateLimiter:
    def __init__(self, max_requests: int, window_seconds: int):
        """
        Initialize the rate limiter.

        Args:
            max_requests: Maximum number of requests allowed per window
            window_seconds: Time window in seconds
        """
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self.user_requests = {}  # Dictionary: user_id -> list of timestamps

    def is_allowed(self, user_id: str) -> bool:
        """
        Check if a request from this user should be allowed.

        Args:
            user_id: Unique identifier for the user

        Returns:
            True if request is allowed, False if rate limited
        """
        current_time = time.time()

        # Step 1: If user is new, create empty list for them
        if user_id not in self.user_requests:
            self.user_requests[user_id] = []

        # Step 2: Get this user's request history
        requests = self.user_requests[user_id]

        # Step 3: Calculate the cutoff time (start of current window)
        window_start = current_time - self.window_seconds

        # Step 4: Remove old requests (keep only those within the window)
        # Filter: keep timestamps that are newer than window_start
        requests = [timestamp for timestamp in requests if timestamp > window_start]
        self.user_requests[user_id] = requests

        # Step 5: Check if user is under the limit
        if len(requests) < self.max_requests:
            # Allowed! Record this request
            requests.append(current_time)
            return True
        else:
            # Rate limited!
            return False

## Step-by-Step Explanation

### The Data Structure

"""
This dictionary maps each user to their list of request timestamps
{
    "user_A": [1705100000.1, 1705100030.5, 1705100045.2],
    "user_B": [1705100010.0]
}

### Walking Through an Example

Let's trace through with `max_requests=3` and `window_seconds=60`:
Time
0: 00 - user_A
makes
request
├── current_time = 0
├── user_requests["user_A"] = [](new
user)
├── window_start = 0 - 60 = -60
├── After
filtering: [](nothing
to
filter)
├── len([]) = 0 < 3? YES
├── Add
timestamp: [0]
└── Return
True ✅

Time
0: 10 - user_A
makes
request
├── current_time = 10
├── user_requests["user_A"] = [0]
├── window_start = 10 - 60 = -50
├── After
filtering: [0](0 > -50, so
keep
it)
├── len([0]) = 1 < 3? YES
├── Add
timestamp: [0, 10]
└── Return
True ✅

Time
0: 20 - user_A
makes
request
├── current_time = 20
├── user_requests["user_A"] = [0, 10]
├── window_start = 20 - 60 = -40
├── After
filtering: [0, 10]
├── len([0, 10]) = 2 < 3? YES
├── Add
timestamp: [0, 10, 20]
└── Return
True ✅

Time
0: 30 - user_A
makes
request
├── current_time = 30
├── user_requests["user_A"] = [0, 10, 20]
├── window_start = 30 - 60 = -30
├── After
filtering: [0, 10, 20]
├── len([0, 10, 20]) = 3 < 3? NO
└── Return
False ❌ (RATE LIMITED!)

Time
1: 05 - user_A
makes
request(65
seconds
later)
├── current_time = 65
├── user_requests["user_A"] = [0, 10, 20]
├── window_start = 65 - 60 = 5
├── After
filtering: [10, 20](0 is removed
because
0 < 5)
├── len([10, 20]) = 2 < 3? YES
├── Add
timestamp: [10, 20, 65]
└── Return
True ✅ (window has moved!)

## Visual Representation
Timeline(seconds):
0
10
20
30
40
50
60
70
| ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- |
▲         ▲         ▲         ▲                                  ▲
req1
req2
req3
req4
req5
✅        ✅        ✅        ❌                                  ✅
(limit reached)(window
moved)

At
time
30:
├── Window
covers: -30
to
30
├── Requests in window: [0, 10, 20] = 3
requests
└── BLOCKED!

At
time
65:
├── Window
covers: 5
to
65
├── Requests in window: [10, 20] = 2
requests(0
fell
out!)
└── ALLOWED!



## Testing the Solution
# Test it yourself
limiter = RateLimiter(max_requests=3, window_seconds=60)

print(limiter.is_allowed("user_A"))  # True
print(limiter.is_allowed("user_A"))  # True
print(limiter.is_allowed("user_A"))  # True
print(limiter.is_allowed("user_A"))  # False (rate limited!)
print(limiter.is_allowed("user_B"))  # True (different user)
## Complexity Analysis

| Aspect | Complexity |
| -------- | ------------ |
| ** Time ** | O(n)
per
request, where
n = requests in window |
| ** Space ** | O(u × r), where
u = users, r = max
requests
stored |


## Follow-Up Questions an Interviewer Might Ask

### 1. "How would you improve the time complexity?"

** Answer: ** Use
a ** deque ** (double - ended queue)
instead
of
a
list.Old
requests
are
always
at
the
front, so
we
can
pop
from the left in O(1):
from collections import deque

# Instead of filtering the whole list:
while requests and requests[0] <= window_start:
    requests.popleft()  # O(1) operation

### 2. "How would this work in a distributed system?"

** Answer: ** Use ** Redis **
with sorted sets:
    - Store
    timestamps in a
    sorted
    set
    per
    user
- Use
`ZREMRANGEBYSCORE`
to
remove
old
entries
- Use
`ZCARD`
to
count
entries
- Redis
handles
concurrency
for you

### 3. "What about memory cleanup for inactive users?"

** Answer: ** Add
a
background
task
that
periodically
removes
users
with no recent requests, or use a TTL-based cache.

## Key Concepts to Remember

1. ** Sliding
Window ** - The
time
window
"slides"
forward
with time
    2. ** Per - User
    Tracking ** - Each
    user
    has
    independent
    limits
3. ** Timestamp
Storage ** - We
store
WHEN
requests
happened, not just
count
4. ** Cleanup ** - Remove
old
timestamps
to
save
memory and maintain
accuracy
## Ready for the Next Challenge?

Would
you
like
to:
1. ** Try
another
coding
problem ** (maybe API design or a data processing challenge)?
2. ** Practice
the
system
design
discussion ** (designing a RAG system)?
3. ** Do
a
mock
behavioral
question
round **?

Let
me
know! 💪
"""