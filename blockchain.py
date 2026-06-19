"""
blockchain.py -- Pure blockchain primitives for Lab 3. No networking.

Import this from lab3.py. All byte orders match the spec exactly:
  Header timestamp : uint64 big-endian, 8 bytes  (struct '>Q')
  Header difficulty: uint32 big-endian, 4 bytes  (struct '>I')
  Header nonce     : uint64 big-endian, 8 bytes  (struct '>Q')
  Tx timestamp (in hash): signed int64 BE, 8 bytes (struct '>q', matching wire type 'q')
"""

import hashlib
import struct
from dataclasses import dataclass, field


# ---------------------------------------------------------------------------
# Primitive hash / pack functions
# ---------------------------------------------------------------------------

def pack_header(
    prev_hash: bytes,
    txs_hash: bytes,
    timestamp: int,
    difficulty: int,
    nonce: int,
) -> bytes:
    """Pack the 84-byte block header in the exact wire order defined by the spec.

    Layout (total 84 bytes):
        prev_hash   32 bytes
        txs_hash    32 bytes
        timestamp    8 bytes  uint64 big-endian
        difficulty   4 bytes  uint32 big-endian
        nonce        8 bytes  uint64 big-endian
    """
    return struct.pack(">32s32sQIQ", prev_hash, txs_hash, timestamp, difficulty, nonce)


def block_hash(header: bytes) -> bytes:
    """Return SHA256(header_bytes). The header must be exactly 84 bytes."""
    return hashlib.sha256(header).digest()


def meets_pow(hash_bytes: bytes, difficulty: int) -> bool:
    """Return True if hash_bytes begins with at least `difficulty` zero bits.

    Splits difficulty into full zero-bytes and a remaining partial-byte mask
    so the check works for any integer difficulty, not just multiples of 8.
    """
    full_bytes, rem_bits = divmod(difficulty, 8)
    if any(b != 0 for b in hash_bytes[:full_bytes]):
        return False
    if rem_bits == 0:
        return True
    # The top rem_bits of the next byte must all be zero.
    mask = (0xFF << (8 - rem_bits)) & 0xFF
    return (hash_bytes[full_bytes] & mask) == 0


def mine_block(
    prev_hash: bytes,
    txs_hash: bytes,
    timestamp: int,
    difficulty: int,
) -> tuple[int, bytes]:
    """Search for a nonce satisfying the PoW condition. Returns (nonce, block_hash).

    Runs synchronously. In async contexts use asyncio.to_thread(mine_block, ...)
    so the event loop is not blocked during the search.
    """
    nonce = 0
    while True:
        header = pack_header(prev_hash, txs_hash, timestamp, difficulty, nonce)
        bh = hashlib.sha256(header).digest()
        if meets_pow(bh, difficulty):
            return nonce, bh
        nonce += 1


def compute_tx_hash(
    sender_key: bytes,
    data: bytes,
    timestamp: int,
    signature: bytes,
) -> bytes:
    """Compute a transaction hash as defined by the spec.

    tx_hash = SHA256(sender_key || data || timestamp_8byte_be || signature)

    The timestamp is encoded as a signed 64-bit big-endian integer (wire type
    'q') to match how it arrives in the SubmitTransaction message.
    """
    ts_bytes = struct.pack(">q", timestamp)
    return hashlib.sha256(sender_key + data + ts_bytes + signature).digest()


def compute_txs_hash(tx_hashes: list[bytes]) -> bytes:
    """Compute the body commitment stored in the block header.

    txs_hash = SHA256(tx_hash_1 || tx_hash_2 || ... || tx_hash_n)

    An empty block must use SHA256(b""), which is NOT 32 zero bytes.
    Passing an empty list correctly produces SHA256(b"").
    """
    return hashlib.sha256(b"".join(tx_hashes)).digest()


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class Transaction:
    """A single transaction as stored in the mempool and in mined blocks."""
    sender_key: bytes   # raw IPv8 public key bytes (length varies by key type)
    data: bytes         # arbitrary payload supplied by the sender
    timestamp: int      # Unix timestamp, signed int64 (wire type 'q')
    signature: bytes    # covers sender_key + data + timestamp_8byte_be
    tx_hash: bytes      # 32-byte SHA256; compute once via compute_tx_hash()


@dataclass
class BlockHeader:
    """All header fields as they appear in the BlockResponse wire message."""
    height: int         # chain height; genesis = 0
    prev_hash: bytes    # 32 bytes; all-zeros for the genesis block
    txs_hash: bytes     # 32 bytes; body commitment over ordered tx hashes
    timestamp: int      # uint64 Unix timestamp
    difficulty: int     # declared PoW difficulty in leading zero bits
    nonce: int          # uint64 PoW nonce found during mining
    block_hash: bytes   # 32 bytes; SHA256 of the packed 84-byte header


@dataclass
class Block:
    """A full block: parsed header plus the ordered list of transactions."""
    header: BlockHeader
    transactions: list[Transaction] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Genesis block (shared constant -- all 3 nodes must boot with this exact block)
# ---------------------------------------------------------------------------


_GENESIS_PREV_HASH  = b"\x00" * 32
_GENESIS_TXSH       = compute_txs_hash([])  # SHA256(b"")
_GENESIS_TIMESTAMP  = 1748217600            # 2025-05-26 00:00:00 UTC
_GENESIS_DIFFICULTY = 1

_genesis_nonce, _genesis_bh = mine_block(
    _GENESIS_PREV_HASH,
    _GENESIS_TXSH,
    _GENESIS_TIMESTAMP,
    _GENESIS_DIFFICULTY,
)

GENESIS_BLOCK = Block(
    header=BlockHeader(
        height=0,
        prev_hash=_GENESIS_PREV_HASH,
        txs_hash=_GENESIS_TXSH,
        timestamp=_GENESIS_TIMESTAMP,
        difficulty=_GENESIS_DIFFICULTY,
        nonce=_genesis_nonce,
        block_hash=_genesis_bh,
    ),
    transactions=[],
)


# ---------------------------------------------------------------------------
# Blockchain
# ---------------------------------------------------------------------------

class Blockchain:
    """In-memory blockchain with a pending transaction mempool.

    The canonical chain is stored as a flat list indexed by height.
    add_block only extends the current tip; fork detection and switching
    are handled at the network layer in BlockchainCommunity (lab3.py).
    """

    def __init__(self) -> None:
        # Pre-seeded with genesis so self.height is always >= 0.
        self._chain: list[Block] = [GENESIS_BLOCK]
        # Mempool keyed by tx_hash for O(1) duplicate detection.
        self._mempool: dict[bytes, Transaction] = {}
        # Set of tx_hashes confirmed in any appended block; prevents re-adding.
        self._confirmed: set[bytes] = set()

    # ------------------------------------------------------------------
    # Accessors
    # ------------------------------------------------------------------

    @property
    def tip(self) -> BlockHeader:
        """Header of the most recently appended block."""
        return self._chain[-1].header

    @property
    def height(self) -> int:
        """Current chain height. Genesis = 0."""
        return self._chain[-1].header.height

    def get_block(self, height: int) -> Block | None:
        """Return the block at `height`, or None if out of range."""
        if 0 <= height < len(self._chain):
            return self._chain[height]
        return None

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def validate_block(self, block: Block, parent: BlockHeader) -> bool:
        """Return True if `block` is structurally valid relative to `parent`.

        Checks performed (in order):
          1. height is exactly parent.height + 1
          2. prev_hash equals parent.block_hash
          3. block_hash matches SHA256 of the packed 84-byte header
          4. block_hash satisfies declared difficulty (leading zero bits)
          5. txs_hash matches SHA256 of concatenated transaction hashes
        """
        h = block.header

        if h.height != parent.height + 1:
            return False

        if h.prev_hash != parent.block_hash:
            return False

        packed = pack_header(h.prev_hash, h.txs_hash, h.timestamp, h.difficulty, h.nonce)
        if h.block_hash != hashlib.sha256(packed).digest():
            return False

        if not meets_pow(h.block_hash, h.difficulty):
            return False

        recomputed = compute_txs_hash([tx.tx_hash for tx in block.transactions])
        if h.txs_hash != recomputed:
            return False

        return True

    # ------------------------------------------------------------------
    # Chain extension
    # ------------------------------------------------------------------

    def add_block(self, block: Block) -> bool:
        """Validate `block` against the current tip and append it if valid.

        Returns True on success. Returns False when:
          - block height does not extend the current tip
          - any validation check in validate_block fails
        On success, all block transactions are moved out of the mempool.
        """
        if block.header.height != self.height + 1:
            return False
        parent = self._chain[-1].header
        if not self.validate_block(block, parent):
            return False
        self._chain.append(block)
        self.remove_txs({tx.tx_hash for tx in block.transactions})
        return True

    def replace_chain(self, new_chain: list[Block]) -> bool:
        """Replace our chain with new_chain if it's valid and strictly longer.

        new_chain must start with the shared genesis block. Every subsequent
        block is validated against its predecessor. On success the mempool is
        updated: transactions confirmed in the old chain but not in the new
        chain are re-added; everything confirmed in the new chain is removed.
        """
        if len(new_chain) <= len(self._chain):
            return False
        if not new_chain or new_chain[0].header.block_hash != GENESIS_BLOCK.header.block_hash:
            return False
        for i in range(1, len(new_chain)):
            if not self.validate_block(new_chain[i], new_chain[i - 1].header):
                return False
        old_chain = self._chain
        self._chain = list(new_chain)
        new_confirmed = {
            tx.tx_hash
            for b in self._chain
            for tx in b.transactions
            if tx.sender_key  # skip placeholder-only Transaction objects
        }
        for old_block in old_chain[1:]:
            for tx in old_block.transactions:
                if tx.sender_key and tx.tx_hash not in new_confirmed:
                    self._mempool[tx.tx_hash] = tx
        self._confirmed = new_confirmed
        self._mempool = {h: tx for h, tx in self._mempool.items() if h not in self._confirmed}
        return True

    # ------------------------------------------------------------------
    # Mempool
    # ------------------------------------------------------------------

    def add_tx(self, tx: Transaction) -> bool:
        """Verify the transaction signature and add it to the mempool.

        The signature must cover sender_key + data + timestamp_8byte_be.
        Returns False if the transaction is already confirmed, already in
        the mempool, or fails signature verification.
        Any crypto exception is caught and treated as rejection.
        """
        if tx.tx_hash in self._confirmed or tx.tx_hash in self._mempool:
            return False

        try:
            from ipv8.keyvault.crypto import default_eccrypto
            pub_key = default_eccrypto.key_from_public_bin(tx.sender_key)
            ts_bytes = struct.pack(">q", tx.timestamp)
            msg = tx.sender_key + tx.data + ts_bytes
            if not default_eccrypto.is_valid_signature(pub_key, msg, tx.signature):
                return False
        except Exception:
            # Malformed key or signature bytes -- reject silently.
            return False

        self._mempool[tx.tx_hash] = tx
        return True

    def pending_txs(self) -> list[Transaction]:
        """Return all mempool transactions in insertion order."""
        return list(self._mempool.values())

    def remove_txs(self, hashes: set[bytes]) -> None:
        """Move the given transaction hashes from the mempool to the confirmed set."""
        for h in hashes:
            self._mempool.pop(h, None)
            self._confirmed.add(h)
