# go-etchash

Etchash go module intended for use by core-pool (and open-ethereum-pool).

* for core-pool see https://github.com/etclabscore/open-etc-pool
* for more information on etchash see https://github.com/eth-classic/etchash
* supports etchash, ethash & ubqhash

### usage (etchash)

```go
var ecip1099FBlockClassic uint64 = 11700000 // classic mainnet
var ecip1099FBlockMordor uint64 = 2520000 // mordor testnet

var hasher = etchash.New(&ecip1099FBlockMordor, nil)

if hasher.Verify(block) {
    ...
}
```

### usage (ethash)

```go
var hasher = etchash.New(nil, nil)

if hasher.Verify(block) {
    ...
}
```

### usage (ubqhash)

```go
var uip1FEpoch uint64 = 22 // ubiq mainnet

var hasher = etchash.New(nil, &uip1FEpoch)

if hasher.Verify(block) {
    ...
}
```

