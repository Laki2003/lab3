"""
lab3.py -- Lab 3 IPv8 communities for the PoW blockchain.

Two communities share a single IPv8 instance:
  RegistrationCommunity -- contacts the Lab 3 server to register the blockchain.
  BlockchainCommunity   -- runs the chain: mining, peer propagation, server queries.

All wire field types match the spec tables in ass3.md exactly.
"""

import asyncio
import hashlib
import time
from dataclasses import dataclass

from ipv8.community import Community, CommunitySettings
from ipv8.configuration import ConfigBuilder, Strategy, WalkerDefinition, default_bootstrap_defs
from ipv8.lazy_community import lazy_wrapper
from ipv8.messaging.payload_dataclass import DataClassPayload
from ipv8.peer import Peer
from ipv8.util import run_forever
from ipv8_service import IPv8

from blockchain import (
    Block,
    BlockHeader,
    Blockchain,
    Transaction,
    compute_tx_hash,
    compute_txs_hash,
    pack_header,
    GENESIS_BLOCK,
    meets_pow,
)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

REGISTRATION_COMMUNITY_ID = bytes.fromhex("4c616233426c6f636b636861696e323032365057")

# Server public key on the registration community. Filter peers by this.
SERVER_REG_KEY_HEX = (
    "4c69624e61434c504b3ae3fc099fb56ca3b5e1de9a1c843387f2acdbb78b1bd4"
    "350ffde518068a0d246344b10d0d8c355fd0d76873e7d7f7838f3715e025af08f"
    "791324495e083331ce6"
)
SERVER_REG_KEY = bytes.fromhex(SERVER_REG_KEY_HEX)


BLOCKCHAIN_COMMUNITY_ID = bytes.fromhex("288d6ef75f492781e1cc6fd39090d4eb6f68ccc5")
# Group ID from Lab 2 registration.
GROUP_ID = "d5ed1c54c1bb75eb"

# Member public keys in registration order (same as Lab 2).
PKEY_LUKA_HEX = (
    "4c69624e61434c504b3aac6fff0f41a5fff0ccbf41cf9b738dfdf008fc2e6ad33"
    "6e9a8003522693a7d5072b2e20cb1434b0d597b2f6ae55544f173edd35775b34d"
    "f92376bb5e995b789d"
)
PKEY_ARNELIS_HEX = (
    "4c69624e61434c504b3a71380741a32f619ff5fbb64a8bace03e4fed5d2d1c2c3"
    "ebb3ef2a83c16eebb3bf7cf3da429621839cfe3c274ac96c99e6e31f010ec813b"
    "9b7cfec63ade8ad5ac"
)
PKEY_LAZAR_HEX = (
    "4c69624e61434c504b3aab6cb54713c99dd236e146c49f32417ec91419abfd40d"
    "89429c3842eb9c8f91bb2da97008ac30ff9ffbabaa5400bce2709901742a47899"
    "bb06d45e9b816321c4"
)

PKEY_LUKA    = bytes.fromhex(PKEY_LUKA_HEX.replace("\n", ""))
PKEY_ARNELIS = bytes.fromhex(PKEY_ARNELIS_HEX.replace("\n", ""))
PKEY_LAZAR   = bytes.fromhex(PKEY_LAZAR_HEX.replace("\n", ""))

MEMBER_KEYS      = [PKEY_LUKA, PKEY_ARNELIS, PKEY_LAZAR]
MEMBER_KEY_HEXES = [
    "4c69624e61434c504b3aac6fff0f41a5fff0ccbf41cf9b738dfdf008fc2e6ad336e9a8003522693a7d5072b2e20cb1434b0d597b2f6ae55544f173edd35775b34df92376bb5e995b789d",
    "4c69624e61434c504b3a71380741a32f619ff5fbb64a8bace03e4fed5d2d1c2c3ebb3ef2a83c16eebb3bf7cf3da429621839cfe3c274ac96c99e6e31f010ec813b9b7cfec63ade8ad5ac",
    "4c69624e61434c504b3aab6cb54713c99dd236e146c49f32417ec91419abfd40d89429c3842eb9c8f91bb2da97008ac30ff9ffbabaa5400bce2709901742a47899bb06d45e9b816321c4",
]


MINING_DIFFICULTY = 17

KEY_FILE = "lazar.pem"


# ---------------------------------------------------------------------------
# Registration community payloads  (message IDs 1-2)
# ---------------------------------------------------------------------------

@dataclass
class RegisterBlockchain(DataClassPayload[1]):
    """Any group member sends this to the Lab 3 server to register the chain."""
    group_id: str        # varlenHutf8 -- group ID
    community_id: bytes  # varlenH     -- 20-byte blockchain community ID


@dataclass
class RegisterResponse(DataClassPayload[2]):
    """Server reply to RegisterBlockchain."""
    success: bool  # ?
    message: str   # varlenHutf8


# ---------------------------------------------------------------------------
# Blockchain community payloads -- server-facing  (message IDs 1-6)
# ---------------------------------------------------------------------------

@dataclass
class SubmitTransaction(DataClassPayload[1]):
    """Server submits a test transaction for inclusion in the blockchain."""
    sender_key: bytes  # varlenH -- raw IPv8 public key of the signer
    data: bytes        # varlenH -- arbitrary payload
    timestamp: int     # q       -- Unix timestamp (signed int64)
    signature: bytes   # varlenH -- over sender_key + data + timestamp_8byte_be


@dataclass
class SubmitTransactionResponse(DataClassPayload[2]):
    """Node reply to SubmitTransaction."""
    success: bool   
    tx_hash: bytes  # varlenH -- 32-byte transaction hash
    message: str    # varlenHutf8


@dataclass
class GetChainHeight(DataClassPayload[3]):
    """Server queries the node's current chain height."""
    request_id: int  # q -- echo'd back so the server can match responses


@dataclass
class ChainHeightResponse(DataClassPayload[4]):
    """Node reply to GetChainHeight."""
    request_id: int  # q
    height: int      # q -- current height; genesis = 0
    tip_hash: bytes  # varlenH -- 32-byte hash of the tip block header


@dataclass
class GetBlock(DataClassPayload[5]):
    """Server requests the full block at a specific height."""
    height: int  # q


@dataclass
class BlockResponse(DataClassPayload[6]):
    """Node reply to GetBlock.

    tx_hashes is the raw concatenation of all 32-byte transaction hashes in
    block order. Use b"" for an empty block (the server splits on 32-byte
    boundaries and recomputes txs_hash to verify the body commitment).
    """
    height: int        # q
    prev_hash: bytes   # varlenH
    txs_hash: bytes    # varlenH
    timestamp: int     # q
    difficulty: int    # q  (int64 on the wire per the spec table)
    nonce: int         # q
    block_hash: bytes  # varlenH
    tx_hashes: bytes   # varlenH -- concatenated 32-byte hashes; b"" if empty


# ---------------------------------------------------------------------------
# Blockchain community payloads -- peer-facing 
# ---------------------------------------------------------------------------

@dataclass
class NewBlock(DataClassPayload[10]):
    """Broadcast a newly mined (or relayed) block to teammates."""
    height: int        # q
    prev_hash: bytes   # varlenH
    txs_hash: bytes    # varlenH
    timestamp: int     # q
    difficulty: int    # q
    nonce: int         # q
    block_hash: bytes  # varlenH
    tx_hashes: bytes   # varlenH


@dataclass
class NewTransaction(DataClassPayload[11]):
    """Broadcast a new mempool transaction to teammates."""
    sender_key: bytes  # varlenH
    data: bytes        # varlenH
    timestamp: int     # q
    signature: bytes
    
_ = RegisterBlockchain("", b"")
_ = RegisterResponse(False, "")
_ = SubmitTransaction(b"", b"", 0, b"")
_ = SubmitTransactionResponse(False, b"", "")
_ = GetChainHeight(0)
_ = ChainHeightResponse(0, 0, b"")
_ = GetBlock(0)
_ = BlockResponse(0, b"", b"", 0, 0, 0, b"", b"")
_ = NewBlock(0, b"", b"", 0, 0, 0, b"", b"")
_ = NewTransaction(b"", b"", 0, b"")   # varlenH


# ---------------------------------------------------------------------------
# RegistrationCommunity
# ---------------------------------------------------------------------------

class RegistrationCommunity(Community):
    """Contacts the Lab 3 registration server and registers the blockchain community ID."""
    community_id = REGISTRATION_COMMUNITY_ID

    def __init__(self, settings: CommunitySettings) -> None:
        super().__init__(settings)
        self._server_peer: Peer | None = None
        self._registered = False
        self._last_waiting_peer_count = -1
        self.add_message_handler(RegisterResponse, self.on_register_response)

    def started(self) -> None:
        self.register_task("reg_discover", self._discover_and_register, interval=2.0, delay=1.0)

    async def _discover_and_register(self) -> None:
        peers = self.get_peers()
        for peer in peers:
            if peer.public_key.key_to_bin().hex() == SERVER_REG_KEY_HEX:
                self._server_peer = peer
        my_key_hex = self.my_peer.public_key.key_to_bin().hex()
        discovered_hexes = {peer.public_key.key_to_bin().hex() for peer in self.get_peers()}
        required_team_peers = max(0, len(MEMBER_KEY_HEXES) - 1)
        discovered_team_peers = sum(
            1 for key_hex in MEMBER_KEY_HEXES
            if key_hex != my_key_hex and key_hex in discovered_hexes
        )
 
        if self._server_peer is not None and not self._registered and discovered_team_peers >= required_team_peers:
            await self.send_registration()
            return
 
        if self._server_peer is not None and not self._registered and discovered_team_peers != self._last_waiting_peer_count:
            self._last_waiting_peer_count = discovered_team_peers
            print(
                f"[RegistrationCommunity] Waiting for teammate discovery before registration: "
                f"{discovered_team_peers}/{required_team_peers} peers discovered."
            )

    async def send_registration(self) -> None:
        self.ez_send(self._server_peer, RegisterBlockchain(GROUP_ID, BLOCKCHAIN_COMMUNITY_ID))
        self._registered = True
        print(f"[RegistrationCommunity] Sent registration with community_id={BLOCKCHAIN_COMMUNITY_ID.hex()}")


    @lazy_wrapper(RegisterResponse)
    async def on_register_response(self, peer: Peer, payload: RegisterResponse) -> None:
        print(f"[RegistrationCommunity] Register response: success={payload.success}, message={payload.message!r}")
        if not payload.success:
            print(f"[RegistrationCommunity] WARNING: registration failed -- {payload.message!r} (re-registration is safe)")

    async def re_register(self) -> None:
        """Force a fresh registration to reset the server's automatic retry counter.

        Call this after confirming all 3 nodes are online and reachable.
        """
        self._registered = False
        if self._server_peer is not None:
            await self.send_registration()


# ---------------------------------------------------------------------------
# BlockchainCommunity
# ---------------------------------------------------------------------------

class BlockchainCommunity(Community):
    """Runs the PoW blockchain: mines blocks, propagates them, answers server queries.

    The Blockchain instance (self.chain) is the single source of truth for the
    canonical chain and the pending mempool. All mutation goes through it.
    """
    community_id = BLOCKCHAIN_COMMUNITY_ID

    def __init__(self, settings: CommunitySettings) -> None:
        super().__init__(settings)
        self.chain = Blockchain()
        self._team_peers: list[Peer] = []
        self._seen_block_hashes: set[bytes] = {GENESIS_BLOCK.header.block_hash}
        self._mining_active = False
        self._pending_block_requests: set[int] = set()
        self._orphan_blocks_by_prev: dict[bytes, list[Block]] = {}
        self._fork_buffer: dict[int, Block] = {}
        self._fork_target_height: int = 0
        self.add_message_handler(SubmitTransaction,   self.on_submit_transaction)
        self.add_message_handler(GetChainHeight,      self.on_get_chain_height)
        self.add_message_handler(ChainHeightResponse, self.on_peer_chain_height_response)
        self.add_message_handler(GetBlock,            self.on_get_block)
        self.add_message_handler(NewBlock,            self.on_new_block)
        self.add_message_handler(NewTransaction,      self.on_new_transaction)
        self.add_message_handler(BlockResponse,       self.on_block_response)

    def started(self) -> None:
        self.register_task("bc_discover",  self._discover_peers, interval=1.0,  delay=0.5)
        self.register_task("mining_loop",  self._mining_loop,    interval=0.1,  delay=2.0)
        self.register_task("status_loop",  self._status_loop,    interval=5.0,  delay=3.0)

    # ------------------------------------------------------------------
    # Peer discovery
    # ------------------------------------------------------------------

    async def _discover_peers(self) -> None:
        my_key_hex = self.my_peer.public_key.key_to_bin().hex()
        current_peers = [
            peer for peer in self.get_peers()
            if peer.public_key.key_to_bin().hex() in MEMBER_KEY_HEXES
            and peer.public_key.key_to_bin().hex() != my_key_hex
        ]
        new_peers = [p for p in current_peers if p not in self._team_peers]
        gone_peers = [p for p in self._team_peers if p not in current_peers]
        self._team_peers = current_peers
        if new_peers or gone_peers:
            print(f"[BC] Team peers: {len(self._team_peers)}/2  keys={[p.public_key.key_to_bin().hex()[:8] for p in self._team_peers]}")
        for peer in new_peers:
            self.ez_send(peer, GetChainHeight(request_id=0))

    # ------------------------------------------------------------------
    # Mining loop
    # ------------------------------------------------------------------

    async def _mining_loop(self) -> None:
        if self._mining_active:
            return
        if not self._team_peers:
            return
        self._mining_active = True
        try:
            pending = self.chain.pending_txs()
            txs_hash = compute_txs_hash([tx.tx_hash for tx in pending])
            prev_hash = self.chain.tip.block_hash
            timestamp = int(time.time())
            height = self.chain.height + 1
            nonce = 0

            while True:
                for _ in range(1000):
                    header = pack_header(prev_hash, txs_hash, timestamp, MINING_DIFFICULTY, nonce)
                    bh = hashlib.sha256(header).digest()
                    if meets_pow(bh, MINING_DIFFICULTY):
                        block = Block(
                            header=BlockHeader(
                                height=height,
                                prev_hash=prev_hash,
                                txs_hash=txs_hash,
                                timestamp=timestamp,
                                difficulty=MINING_DIFFICULTY,
                                nonce=nonce,
                                block_hash=bh,
                            ),
                            transactions=pending,
                        )
                        if self.chain.add_block(block):
                            print(f"[MINE] Block #{height:<6}  txs={len(pending)}")
                            await self._propagate_block(block)
                        return
                    nonce += 1

                await asyncio.sleep(0.05)  # yield to event loop — lets NewBlock messages land

                if self.chain.tip.block_hash != prev_hash:
                    return  # a peer won this round; restart on new tip next tick
        finally:
            self._mining_active = False

    # ------------------------------------------------------------------
    # Serialization helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _serialize_tx_hashes(block: Block) -> bytes:
        """Concatenate transaction hashes into the raw bytes expected on the wire.

        Returns b"" for an empty block (not 32 zero bytes).
        """
        return b"".join(tx.tx_hash for tx in block.transactions)

    @staticmethod
    def _block_to_new_block(block: Block) -> NewBlock:
        """Serialize a Block into a NewBlock peer-propagation payload."""
        h = block.header
        return NewBlock(
            height=h.height,
            prev_hash=h.prev_hash,
            txs_hash=h.txs_hash,
            timestamp=h.timestamp,
            difficulty=h.difficulty,
            nonce=h.nonce,
            block_hash=h.block_hash,
            tx_hashes=b"".join(tx.tx_hash for tx in block.transactions),
        )

    @staticmethod
    def _block_to_block_response(block: Block) -> BlockResponse:
        """Serialize a Block into a BlockResponse server-query payload."""
        h = block.header
        return BlockResponse(
            height=h.height,
            prev_hash=h.prev_hash,
            txs_hash=h.txs_hash,
            timestamp=h.timestamp,
            difficulty=h.difficulty,
            nonce=h.nonce,
            block_hash=h.block_hash,
            tx_hashes=b"".join(tx.tx_hash for tx in block.transactions),
        )

    @staticmethod
    def _block_from_new_block(payload: NewBlock) -> Block:
        """Reconstruct a Block from a NewBlock payload.

        NOTE: NewBlock carries only tx_hashes (32-byte identifiers), not full
        transaction bodies. The reconstructed Block has placeholder Transaction
        objects that contain tx_hash only. This is sufficient for chain
        validation (txs_hash recomputation) and for responding to GetBlock
        (the server splits tx_hashes on 32-byte boundaries, not tx bodies).
        If full transaction bodies are needed later, add a separate fetch step.
        """
        # Split the concatenated tx_hashes into 32-byte chunks.
        raw = payload.tx_hashes
        hashes = [raw[i:i + 32] for i in range(0, len(raw), 32)]
        # Build placeholder Transaction objects carrying only the hash.
        txs = [
            Transaction(
                sender_key=b"",
                data=b"",
                timestamp=0,
                signature=b"",
                tx_hash=h,
            )
            for h in hashes
        ]
        header = BlockHeader(
            height=payload.height,
            prev_hash=payload.prev_hash,
            txs_hash=payload.txs_hash,
            timestamp=payload.timestamp,
            difficulty=payload.difficulty,
            nonce=payload.nonce,
            block_hash=payload.block_hash,
        )
        return Block(header=header, transactions=txs)

    # ------------------------------------------------------------------
    # Propagation helpers
    # ------------------------------------------------------------------

    async def _propagate_block(self, block: Block, exclude: Peer | None = None) -> None:
        payload = self._block_to_new_block(block)
        self._seen_block_hashes.add(block.header.block_hash)
        for peer in self._team_peers:
            if peer != exclude:
                self.ez_send(peer, payload)
        

    async def _propagate_tx(self, tx: Transaction, exclude: Peer | None = None) -> None:
        transaction_payload = NewTransaction(
            sender_key=tx.sender_key, data=tx.data,
            timestamp=tx.timestamp, signature=tx.signature)
        for peer in self._team_peers:
            if peer != exclude:
                self.ez_send(peer, transaction_payload)

    # ------------------------------------------------------------------
    # Server-facing handlers
    # ------------------------------------------------------------------

    @lazy_wrapper(SubmitTransaction)
    async def on_submit_transaction(self, peer: Peer, payload: SubmitTransaction) -> None:
        tx_hash = compute_tx_hash(payload.sender_key, payload.data, payload.timestamp, payload.signature)
        transaction = Transaction(
            sender_key=payload.sender_key, data=payload.data,
            timestamp=payload.timestamp, signature=payload.signature, tx_hash=tx_hash)
        success = self.chain.add_tx(transaction)
        if success:
            await self._propagate_tx(transaction)
        message = "Transaction accepted." if success else "Transaction rejected: invalid signature or duplicate."
        self.ez_send(peer, 
            SubmitTransactionResponse(success=success, tx_hash=tx_hash, message=message))
        

    async def _status_loop(self) -> None:
        # Re-request any blocks that never arrived during an in-progress fork sync.
        # Without this, a single dropped UDP packet freezes sync permanently.
        if self._fork_target_height > 0 and self._team_peers:
            missing = [h for h in range(1, self._fork_target_height + 1) if h not in self._fork_buffer]
            if missing:
                peer = self._team_peers[0]
                print(f"[SYNC]  Retrying {len(missing)} missing block(s) up to height {self._fork_target_height}")
                for h in missing:
                    self.ez_send(peer, GetBlock(h))

        peer_key = self.my_peer.public_key.key_to_bin().hex()[:12]
        tip_hash = self.chain.tip.block_hash.hex()
        print(f"[STATUS] local height={self.chain.height}  tip_hash={tip_hash}  peers={len(self._team_peers)}  (key={peer_key}...)")
        for peer in self._team_peers:
            self.ez_send(peer, GetChainHeight(request_id=1))

    @lazy_wrapper(ChainHeightResponse)
    async def on_peer_chain_height_response(self, peer: Peer, payload: ChainHeightResponse) -> None:
        peer_key = peer.public_key.key_to_bin().hex()[:12]
        print(f"[STATUS] peer  height={payload.height}  (key={peer_key}...)")
        if payload.height > self.chain.height and (not self._fork_target_height or payload.height > self._fork_target_height):
            await self._trigger_fork_sync(peer, payload.height)

    @lazy_wrapper(GetChainHeight)
    async def on_get_chain_height(self, peer: Peer, payload: GetChainHeight) -> None:
        self.ez_send(peer , ChainHeightResponse(
                   payload.request_id,
                   self.chain.height,
                   self.chain.tip.block_hash)
                   )


    @lazy_wrapper(GetBlock)
    async def on_get_block(self, peer: Peer, payload: GetBlock) -> None:
        block = self.chain.get_block(payload.height)
        if block is not None:
            self.ez_send(peer, self._block_to_block_response(block))
        else:
            print("Block not found.")

    # ------------------------------------------------------------------
    # Peer-facing handlers
    # ------------------------------------------------------------------

    @lazy_wrapper(NewBlock)
    async def on_new_block(self, peer: Peer, payload: NewBlock) -> None:
        if payload.block_hash in self._seen_block_hashes:
            return
        self._seen_block_hashes.add(payload.block_hash)
        block = self._block_from_new_block(payload)
        added = self.chain.add_block(block)
        if added:
            print(f"[RECV] Block #{payload.height:<6}  from peer (key={peer.public_key.key_to_bin().hex()})")
            self._adopt_orphans_from_tip()
            await self._propagate_block(block, exclude=peer)
        elif payload.height > self.chain.height + 1:
            self._store_orphan_block(block)
            if not self._fork_target_height:
                await self._trigger_fork_sync(peer, payload.height)
            for h in range(self.chain.height + 1, payload.height):
                if h not in self._pending_block_requests:
                    self._pending_block_requests.add(h)
                    self.ez_send(peer, GetBlock(h))
        elif payload.height > self.chain.height:
            # height == tip+1 but wrong prev_hash: fork right at the tip
            if not self._fork_target_height:
                await self._trigger_fork_sync(peer, payload.height)

    @lazy_wrapper(NewTransaction)
    async def on_new_transaction(self, peer: Peer, payload: NewTransaction) -> None:
        tx_hash = compute_tx_hash(payload.sender_key, payload.data, payload.timestamp, payload.signature)
        transaction = Transaction(
            sender_key=payload.sender_key, data=payload.data, timestamp=payload.timestamp, signature=payload.signature, tx_hash=tx_hash)
        accepted = self.chain.add_tx(transaction)
        if accepted:
            await self._propagate_tx(transaction, exclude=peer)

    @lazy_wrapper(BlockResponse)
    async def on_block_response(self, peer: Peer, payload: BlockResponse) -> None:
        """Handle a block body received in response to a GetBlock sync request."""
        self._pending_block_requests.discard(payload.height)
        raw = payload.tx_hashes
        tx_hashes = [raw[i:i + 32] for i in range(0, len(raw), 32)]
        block = Block(
            header=BlockHeader(
                height=payload.height,
                prev_hash=payload.prev_hash,
                txs_hash=payload.txs_hash,
                timestamp=payload.timestamp,
                difficulty=payload.difficulty,
                nonce=payload.nonce,
                block_hash=payload.block_hash,
            ),
            transactions=[
                Transaction(sender_key=b"", data=b"", timestamp=0, signature=b"", tx_hash=h)
                for h in tx_hashes
            ],
        )
        # Always feed the fork buffer while a fork sync is in progress,
        # even for blocks already seen via NewBlock.
        if self._fork_target_height > 0:
            self._fork_buffer[payload.height] = block
            await self._try_apply_fork()

        if payload.block_hash in self._seen_block_hashes:
            return
        self._seen_block_hashes.add(block.header.block_hash)
        if self.chain.add_block(block):
            self._clear_satisfied_requests()
            self._adopt_orphans_from_tip()
        else:
            self._store_orphan_block(block)
            if not self._fork_target_height and payload.height > self.chain.height:
                await self._trigger_fork_sync(peer, payload.height)

    async def _trigger_fork_sync(self, peer: Peer, peer_height: int) -> None:
        """Request the peer's full chain (heights 1..peer_height) to resolve a fork."""
        if peer_height < self._fork_target_height:
            return
        print(f"[SYNC] Catching up   local={self.chain.height} -> peer={peer_height}  ({peer_height - self.chain.height} blocks)")
        self._fork_buffer = {}
        self._fork_target_height = peer_height
        for h in range(1, peer_height + 1):
            self.ez_send(peer, GetBlock(h))

    async def _try_apply_fork(self) -> None:
        """If we have all blocks 1..fork_target_height, attempt to switch chains."""
        if not self._fork_target_height:
            return
        if not all(h in self._fork_buffer for h in range(1, self._fork_target_height + 1)):
            return
        candidate = [GENESIS_BLOCK] + [
            self._fork_buffer[h] for h in range(1, self._fork_target_height + 1)
        ]
        if self.chain.replace_chain(candidate):
            print(f"[SYNC] Done          switched to height={self.chain.height}")
            for b in candidate:
                self._seen_block_hashes.add(b.header.block_hash)
            self._adopt_orphans_from_tip()
        else:
            print(f"[SYNC] Rejected      peer chain not longer (our height={self.chain.height})")
        self._fork_buffer.clear()
        self._fork_target_height = 0

    def _clear_satisfied_requests(self) -> None:
        self._pending_block_requests = {h for h in self._pending_block_requests if h > self.chain.height}

    def _store_orphan_block(self, block: Block) -> None:
        """Store blocks that arrive before their parent.
 
        We keep them by parent hash and try again when that parent is appended.
        """
        children = self._orphan_blocks_by_prev.setdefault(block.header.prev_hash, [])
        if any(b.header.block_hash == block.header.block_hash for b in children):
            return
        children.append(block)
 
 
    def _adopt_orphans_from_tip(self) -> None:
        """Append any buffered child blocks that now connect to our tip."""
        while True:
            parent_hash = self.chain.tip.block_hash
            children = self._orphan_blocks_by_prev.pop(parent_hash, [])
            if not children:
                break
            adopted_any = False
            deferred: list[Block] = []
            for child in children:
                if self.chain.add_block(child):
                    self._seen_block_hashes.add(child.header.block_hash)
                    adopted_any = True
                else:
                    deferred.append(child)
            if deferred:
                self._orphan_blocks_by_prev[parent_hash] = deferred
            if not adopted_any:
                break
 
 


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------



async def start_client() -> None:
    builder = ConfigBuilder().clear_keys().clear_overlays()
    builder.add_key("client", "curve25519", KEY_FILE)
    builder.add_overlay(
        "RegistrationCommunity", "client",
        [WalkerDefinition(Strategy.RandomWalk, 20, {"timeout": 3.0})],
        default_bootstrap_defs, {}, [("started",)],
    )
    builder.add_overlay(
        "BlockchainCommunity", "client",
        [WalkerDefinition(Strategy.RandomWalk, 20, {"timeout": 3.0})],
        default_bootstrap_defs, {}, [("started",)],
    )

    ipv8 = IPv8(
        builder.finalize(),
        extra_communities={
            "RegistrationCommunity": RegistrationCommunity,
            "BlockchainCommunity": BlockchainCommunity,
        },
    )
    await ipv8.start()
    await run_forever()

if __name__ == "__main__":
    asyncio.run(start_client())
