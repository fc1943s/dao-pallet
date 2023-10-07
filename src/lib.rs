#![cfg_attr(not(feature = "std"), no_std)]

use frame_support::traits::{Currency, Hooks, OnUnbalanced, Randomness, ReservableCurrency};
use frame_support::PalletId;
pub use pallet::*;
use pallet_timestamp::{self as timestamp};
use sp_runtime::traits::Zero;
use sp_runtime::SaturatedConversion;
use sp_std::prelude::*;
use std::collections::BTreeMap;

type AccountIdOf<T> = <T as frame_system::Config>::AccountId;
type BalanceOf<T> = <<T as Config>::Currency as Currency<AccountIdOf<T>>>::Balance;
type NegativeImbalanceOf<T> =
    <<T as Config>::Currency as Currency<AccountIdOf<T>>>::NegativeImbalance;
type _Hooks = dyn Hooks<()>;

fn get_round_minutes(now: u64, block_timestamp: u64) -> f64 {
    core::time::Duration::from_millis(now - block_timestamp).as_secs_f64() / 60.
}

fn new_hash(value: u64, dummy: u64) -> [u8; 32] {
    let mut entropy: Vec<u8> = Vec::new();
    entropy.extend(value.to_le_bytes());
    entropy.extend(dummy.to_le_bytes());
    sp_io::hashing::keccak_256(&entropy)
}

#[frame_support::pallet]
pub mod pallet {
    use super::*;
    use frame_support::pallet_prelude::*;
    use frame_system::pallet_prelude::*;

    #[pallet::config]
    pub trait Config: frame_system::Config + timestamp::Config {
        type RuntimeEvent: From<Event<Self>> + IsType<<Self as frame_system::Config>::RuntimeEvent>;

        type Currency: ReservableCurrency<Self::AccountId>;

        #[pallet::constant]
        type PalletId: Get<PalletId>;

        type Randomness: Randomness<Self::Hash, BlockNumberFor<Self>>;

        type Slashed: OnUnbalanced<NegativeImbalanceOf<Self>>;

        type ForceOrigin: EnsureOrigin<Self::RuntimeOrigin>;

        #[pallet::constant]
        type JoinFee: Get<BalanceOf<Self>>;

        #[pallet::constant]
        type SubmitFee: Get<BalanceOf<Self>>;

        #[pallet::constant]
        type RoundMinutes: Get<u64>;
    }

    #[derive(Encode, Decode, PartialEq, TypeInfo, MaxEncodedLen)]
    pub enum MemberType {
        Participant,
        Bot,
    }

    #[derive(Encode, Decode, PartialEq, TypeInfo, MaxEncodedLen, Default)]
    pub enum RoundType {
        #[default]
        Submission,
        Reveal,
    }

    #[pallet::event]
    #[pallet::generate_deposit(pub(super) fn deposit_event)]
    pub enum Event<T: Config> {
        NumberGenerated { hash: String },
    }

    #[pallet::error]
    pub enum Error<T> {
        AlreadyJoined,
        InvalidRound,
        RoundNotFinished,
        ParticipantRequired,
        BotRequired,
        InvalidHash,
        AlreadySubmitted,
        NotSubmitted,
        AlreadyRevealed,
    }

    #[pallet::storage]
    pub type Members<T: Config> = StorageMap<_, Twox64Concat, T::AccountId, MemberType>;

    #[pallet::storage]
    pub type Hashes<T: Config> =
        StorageMap<_, Twox64Concat, T::AccountId, BoundedVec<u8, ConstU32<64>>>;

    #[pallet::storage]
    pub type Values<T: Config> = StorageMap<_, Twox64Concat, T::AccountId, (u64, Option<u64>)>;

    #[pallet::storage]
    pub type Round<T> = StorageValue<_, RoundType, ValueQuery>;

    #[pallet::storage]
    pub type RoundTimestamp<T> = StorageValue<
        _,
        <pallet_timestamp::Pallet<T> as frame_support::traits::Time>::Moment,
        ValueQuery,
    >;

    #[pallet::pallet]
    pub struct Pallet<T>(_);

    #[pallet::call]
    impl<T: Config> Pallet<T> {
        #[pallet::call_index(0)]
        #[pallet::weight({70_000_000})]
        pub fn join(origin: OriginFor<T>, is_bot: bool) -> DispatchResult {
            let sender = ensure_signed(origin)?;

            ensure!(
                !<Members<T>>::contains_key(&sender),
                Error::<T>::AlreadyJoined
            );

            T::Currency::reserve(&sender, T::JoinFee::get())?;

            <Members<T>>::insert(
                &sender,
                if is_bot {
                    MemberType::Bot
                } else {
                    MemberType::Participant
                },
            );

            Ok(())
        }

        #[pallet::call_index(1)]
        #[pallet::weight({70_000_000})]
        pub fn submit_hash(origin: OriginFor<T>, hash: String) -> DispatchResult {
            let sender = ensure_signed(origin)?;

            ensure!(
                <Round<T>>::get() == RoundType::Submission,
                Error::<T>::InvalidRound
            );

            let round_timestamp = <RoundTimestamp<T>>::get();
            if round_timestamp.is_zero() {
                <RoundTimestamp<T>>::put(<timestamp::Pallet<T>>::get());
            } else {
                let round_minutes = get_round_minutes(
                    timestamp::Pallet::<T>::get().saturated_into::<u64>(),
                    round_timestamp.saturated_into::<u64>(),
                );
                ensure!(
                    round_minutes < T::RoundMinutes::get() as f64,
                    Error::<T>::InvalidRound
                );
            }

            let member = <Members<T>>::get(&sender);
            ensure!(
                member == Some(MemberType::Participant),
                Error::<T>::ParticipantRequired
            );

            ensure!(
                !<Hashes<T>>::contains_key(&sender),
                Error::<T>::AlreadySubmitted
            );

            T::Currency::reserve(&sender, T::SubmitFee::get())?;

            let bounded_hash: BoundedVec<_, _> = hex::decode(hash)
                .unwrap()
                .try_into()
                .map_err(|_| Error::<T>::InvalidHash)?;

            <Hashes<T>>::insert(&sender, bounded_hash);

            Ok(())
        }

        #[pallet::call_index(2)]
        #[pallet::weight({70_000_000})]
        pub fn submit_value(origin: OriginFor<T>, value: u64) -> DispatchResult {
            let sender = ensure_signed(origin)?;

            let member = <Members<T>>::get(&sender);
            ensure!(member == Some(MemberType::Bot), Error::<T>::BotRequired);

            match <Round<T>>::get() {
                RoundType::Submission
                    if get_round_minutes(
                        timestamp::Pallet::<T>::get().saturated_into::<u64>(),
                        <RoundTimestamp<T>>::get().saturated_into::<u64>(),
                    ) >= T::RoundMinutes::get() as f64 =>
                {
                    <Round<T>>::put(RoundType::Reveal)
                }
                RoundType::Submission => ensure!(false, Error::<T>::RoundNotFinished),
                RoundType::Reveal => {}
            }

            ensure!(
                !<Values<T>>::contains_key(&sender),
                Error::<T>::AlreadySubmitted
            );

            <Values<T>>::insert(&sender, (value, None::<u64>));

            Ok(())
        }

        #[pallet::call_index(3)]
        #[pallet::weight({70_000_000})]
        pub fn reveal(origin: OriginFor<T>, value: u64, dummy: u64) -> DispatchResult {
            let sender = ensure_signed(origin)?;

            match <Round<T>>::get() {
                RoundType::Submission
                    if get_round_minutes(
                        timestamp::Pallet::<T>::get().saturated_into::<u64>(),
                        <RoundTimestamp<T>>::get().saturated_into::<u64>(),
                    ) >= T::RoundMinutes::get() as f64 =>
                {
                    <Round<T>>::put(RoundType::Reveal)
                }
                RoundType::Submission => ensure!(false, Error::<T>::RoundNotFinished),
                RoundType::Reveal => {}
            }

            let member = <Members<T>>::get(&sender);
            ensure!(
                member == Some(MemberType::Participant),
                Error::<T>::ParticipantRequired
            );

            ensure!(<Hashes<T>>::contains_key(&sender), Error::<T>::NotSubmitted);

            ensure!(
                !<Values<T>>::contains_key(&sender),
                Error::<T>::AlreadyRevealed
            );

            T::Currency::reserve(&sender, T::SubmitFee::get())?;

            <Values<T>>::insert(&sender, (value, Some(dummy)));

            Ok(())
        }

        #[pallet::call_index(8)]
        #[pallet::weight(({70_000_000}, DispatchClass::Operational))]
        pub fn generate_random_number(origin: OriginFor<T>) -> DispatchResult {
            T::ForceOrigin::ensure_origin(origin)?;

            ensure!(
                <Round<T>>::get() == RoundType::Reveal,
                Error::<T>::RoundNotFinished
            );

            let data: BTreeMap<
                T::AccountId,
                (
                    MemberType,
                    BalanceOf<T>,
                    Option<BoundedVec<_, _>>,
                    (u64, Option<u64>),
                ),
            > = <Values<T>>::iter()
                .map(|(account_id, value)| {
                    let member_type = <Members<T>>::get(account_id.clone()).unwrap();
                    let reserved = T::Currency::reserved_balance(&account_id);
                    let hash = <Hashes<T>>::get(account_id.clone());
                    (account_id, (member_type, reserved, hash, value))
                })
                .collect();

            let participant_values: BTreeMap<T::AccountId, (BalanceOf<T>, Option<u64>)> = data
                .iter()
                .filter(|(_, (_, _, _, (_, dummy)))| dummy.is_some())
                .map(|(account_id, (_, reserved, stored_hash, (value, dummy)))| {
                    let hash_u8 = new_hash(*value, dummy.unwrap());
                    let hash = hash_u8.to_vec();

                    let result = if stored_hash.as_ref().unwrap().to_vec() == hash {
                        Some(*value)
                    } else {
                        None
                    };
                    (account_id.clone(), (*reserved, result))
                })
                .collect();

            let (valid_participant_values, invalid_participant_values): (
                BTreeMap<T::AccountId, (BalanceOf<T>, Option<u64>)>,
                BTreeMap<T::AccountId, (BalanceOf<T>, Option<u64>)>,
            ) = participant_values
                .into_iter()
                .partition(|(_, (_, value))| value.is_some());

            let (valid_participant_values, invalid_participant_values): (
                BTreeMap<T::AccountId, (BalanceOf<T>, u64)>,
                BTreeMap<T::AccountId, BalanceOf<T>>,
            ) = (
                valid_participant_values
                    .into_iter()
                    .map(|(account_id, (reserved, value))| (account_id, (reserved, value.unwrap())))
                    .collect(),
                invalid_participant_values
                    .into_iter()
                    .map(|(account_id, (reserved, _))| (account_id, reserved))
                    .collect(),
            );

            let valid_participant_count = valid_participant_values.len();

            let max_bot_count = if valid_participant_count == 0 {
                data.len()
            } else {
                (valid_participant_count as f64 * 0.25) as usize
            };

            let bot_values: BTreeMap<T::AccountId, u64> = data
                .iter()
                .filter(|(_, (_, _, _, (_, dummy)))| dummy.is_none())
                .map(|(account_id, (_, _, _, (value, _)))| (account_id.clone(), *value))
                .take(max_bot_count)
                .collect();

            let mut entropy = valid_participant_values
                .iter()
                .map(|(_, (_, value))| value)
                .chain(bot_values.iter().map(|(_, value)| value))
                .map(|value| (*value).to_le_bytes())
                .flatten()
                .collect::<Vec<_>>();

            let (random_seed, _) = T::Randomness::random(&(T::PalletId::get(), 0).encode());
            let seed = format!("{:?}", random_seed);

            entropy.extend(seed.bytes());

            let hash_u8 = sp_io::hashing::keccak_256(&entropy);

            let hash = hex::encode(hash_u8);

            Self::deposit_event(Event::<T>::NumberGenerated { hash });

            valid_participant_values
                .iter()
                .for_each(|(account_id, (_reserved, _))| {
                    let amount = T::SubmitFee::get().saturated_into::<u64>() * 2;
                    T::Currency::unreserve(account_id, amount.saturated_into());
                });

            invalid_participant_values
                .iter()
                .for_each(|(account_id, _reserved)| {
                    let amount = T::SubmitFee::get().saturated_into::<u64>() * 2;
                    T::Slashed::on_unbalanced(
                        T::Currency::slash_reserved(account_id, amount.saturated_into()).0,
                    );
                    T::Currency::unreserve(account_id, amount.saturated_into());
                });

            <Round<T>>::put(RoundType::Submission);
            <RoundTimestamp<T>>::put(0u64.saturated_into::<T::Moment>());

            let _ = <Hashes<T>>::clear(u32::MAX, None);
            let _ = <Values<T>>::clear(u32::MAX, None);

            Ok(())
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate as pallet_nicks;

    use frame_support::{
        assert_noop, assert_ok, ord_parameter_types,
        traits::{ConstU32, ConstU64},
    };
    use frame_system::EnsureSignedBy;
    use sp_core::H256;
    use sp_runtime::{
        traits::{BadOrigin, BlakeTwo256, IdentityLookup},
        BuildStorage,
    };

    type Block = frame_system::mocking::MockBlock<Test>;

    frame_support::construct_runtime!(
        pub enum Test
        {
            System: frame_system,
            Balances: pallet_balances,
            Subject: pallet_nicks,
            Timestamp: pallet_timestamp,
        }
    );

    impl frame_system::Config for Test {
        type BaseCallFilter = frame_support::traits::Everything;
        type BlockWeights = ();
        type BlockLength = ();
        type DbWeight = ();
        type RuntimeOrigin = RuntimeOrigin;
        type Nonce = u64;
        type Hash = H256;
        type RuntimeCall = RuntimeCall;
        type Hashing = BlakeTwo256;
        type AccountId = u64;
        type Lookup = IdentityLookup<Self::AccountId>;
        type Block = Block;
        type RuntimeEvent = RuntimeEvent;
        type BlockHashCount = ConstU64<250>;
        type Version = ();
        type PalletInfo = PalletInfo;
        type AccountData = pallet_balances::AccountData<u64>;
        type OnNewAccount = ();
        type OnKilledAccount = ();
        type SystemWeightInfo = ();
        type SS58Prefix = ();
        type OnSetCode = ();
        type MaxConsumers = ConstU32<16>;
    }

    impl pallet_balances::Config for Test {
        type MaxLocks = ();
        type MaxReserves = ();
        type ReserveIdentifier = [u8; 8];
        type Balance = u64;
        type RuntimeEvent = RuntimeEvent;
        type DustRemoval = ();
        type ExistentialDeposit = ConstU64<1>;
        type AccountStore = System;
        type WeightInfo = ();
        type FreezeIdentifier = ();
        type MaxFreezes = ();
        type RuntimeHoldReason = ();
        type MaxHolds = ();
    }

    ord_parameter_types! {
        pub const One: u64 = 1;
    }

    impl pallet_timestamp::Config for Test {
        type Moment = u64;
        type OnTimestampSet = ();
        type MinimumPeriod = ConstU64<5>;
        type WeightInfo = ();
    }

    pub struct TestRandomness<T>(sp_std::marker::PhantomData<T>);

    impl<Output: codec::Decode + Default, T>
        frame_support::traits::Randomness<Output, frame_system::pallet_prelude::BlockNumberFor<T>>
        for TestRandomness<T>
    where
        T: frame_system::Config,
    {
        fn random(subject: &[u8]) -> (Output, frame_system::pallet_prelude::BlockNumberFor<T>) {
            use sp_runtime::traits::TrailingZeroInput;

            (
                Output::decode(&mut TrailingZeroInput::new(subject)).unwrap_or_default(),
                frame_system::Pallet::<T>::block_number(),
            )
        }
    }

    frame_support::parameter_types! {
        pub const TestPalletId: PalletId = PalletId(*b"PalletId");
    }

    impl Config for Test {
        type RuntimeEvent = RuntimeEvent;
        type Currency = Balances;
        type Slashed = ();
        type ForceOrigin = EnsureSignedBy<One, u64>;

        type Randomness = TestRandomness<Self>;
        type PalletId = TestPalletId;

        type JoinFee = ConstU64<5>;
        type SubmitFee = ConstU64<2>;
        type RoundMinutes = ConstU64<5>;
    }

    fn new_test_ext() -> sp_io::TestExternalities {
        let mut t = frame_system::GenesisConfig::<Test>::default()
            .build_storage()
            .unwrap();
        pallet_balances::GenesisConfig::<Test> {
            balances: vec![(1, 10), (2, 10), (3, 10), (4, 10), (5, 10)],
        }
        .assimilate_storage(&mut t)
        .unwrap();
        t.into()
    }

    pub fn next_block() {
        let block = System::block_number();
        let new_block = block + 1;

        Subject::on_finalize(block);
        Timestamp::on_finalize(block);
        System::on_finalize(block);

        System::set_block_number(new_block);
        frame_system::Pallet::<Test>::set_block_number(new_block);

        System::on_initialize(new_block);
        Timestamp::on_initialize(new_block);
        Subject::on_initialize(new_block);
    }

    pub fn next_minutes() {
        timestamp::Pallet::<Test>::set(
            RuntimeOrigin::none(),
            timestamp::Pallet::<Test>::get() + 1000 * 60 * 5,
        )
        .unwrap();
    }

    fn validate_hashes() -> usize {
        System::events()
            .iter()
            .filter_map(|r| match &r.event {
                RuntimeEvent::Subject(Event::NumberGenerated { hash }) => {
                    assert_eq!(hash.len(), 64);
                    Some(hash)
                }
                _ => None,
            })
            .count()
    }

    #[test]
    fn e2e() {
        new_test_ext().execute_with(|| {
            assert_ok!(Subject::join(RuntimeOrigin::signed(2), false));

            assert_eq!(Balances::reserved_balance(2), 5);
            assert_eq!(Balances::free_balance(2), 5);

            assert_ok!(Subject::join(RuntimeOrigin::signed(3), false));
            assert_noop!(
                Subject::join(RuntimeOrigin::signed(3), false),
                Error::<Test>::AlreadyJoined
            );

            assert_eq!(Balances::reserved_balance(3), 5);
            assert_eq!(Balances::free_balance(3), 5);

            assert_ok!(Subject::join(RuntimeOrigin::signed(4), false));
            assert_ok!(Subject::join(RuntimeOrigin::signed(5), true));

            assert_ok!(Subject::submit_hash(
                RuntimeOrigin::signed(2),
                hex::encode(new_hash(1, 2))
            ));

            assert_eq!(Balances::reserved_balance(2), 7);
            assert_eq!(Balances::free_balance(2), 3);

            assert_ok!(Subject::submit_hash(
                RuntimeOrigin::signed(3),
                hex::encode(new_hash(1, 2))
            ));

            assert_eq!(Balances::reserved_balance(3), 7);
            assert_eq!(Balances::free_balance(3), 3);

            next_minutes();

            assert_ok!(Subject::reveal(RuntimeOrigin::signed(2), 1, 2));

            assert_eq!(Balances::reserved_balance(2), 9);
            assert_eq!(Balances::free_balance(2), 1);

            assert_ok!(Subject::reveal(RuntimeOrigin::signed(3), 3, 4));

            assert_eq!(Balances::reserved_balance(3), 9);
            assert_eq!(Balances::free_balance(3), 1);

            assert_noop!(
                Subject::generate_random_number(RuntimeOrigin::signed(2)),
                BadOrigin
            );
            assert_ok!(Subject::generate_random_number(RuntimeOrigin::signed(1)));

            assert_eq!(Balances::reserved_balance(2), 5);
            assert_eq!(Balances::free_balance(2), 5);

            assert_eq!(Balances::reserved_balance(3), 1);
            assert_eq!(Balances::free_balance(3), 5);

            //

            assert_ok!(Subject::submit_hash(
                RuntimeOrigin::signed(2),
                hex::encode(new_hash(1, 2))
            ));

            assert_noop!(
                Subject::submit_hash(RuntimeOrigin::signed(2), hex::encode(new_hash(1, 2))),
                Error::<Test>::AlreadySubmitted
            );

            assert_noop!(
                Subject::submit_value(RuntimeOrigin::signed(5), 1),
                Error::<Test>::RoundNotFinished
            );

            assert_noop!(
                Subject::submit_hash(RuntimeOrigin::signed(5), hex::encode(new_hash(1, 2))),
                Error::<Test>::ParticipantRequired
            );

            assert_eq!(Balances::reserved_balance(2), 7);
            assert_eq!(Balances::free_balance(2), 3);

            assert_ok!(Subject::submit_hash(
                RuntimeOrigin::signed(3),
                hex::encode(new_hash(1, 2))
            ));

            assert_eq!(Balances::reserved_balance(3), 3);
            assert_eq!(Balances::free_balance(3), 3);

            assert_noop!(
                Subject::reveal(RuntimeOrigin::signed(2), 1, 2),
                Error::<Test>::RoundNotFinished
            );

            assert_noop!(
                Subject::generate_random_number(RuntimeOrigin::signed(1)),
                Error::<Test>::RoundNotFinished
            );

            next_block();
            next_minutes();

            assert_eq!(System::events().len(), 0);

            assert_noop!(
                Subject::submit_hash(RuntimeOrigin::signed(4), hex::encode(new_hash(1, 2))),
                Error::<Test>::InvalidRound
            );

            assert_noop!(
                Subject::submit_value(RuntimeOrigin::signed(2), 1),
                Error::<Test>::BotRequired
            );

            assert_ok!(Subject::submit_value(RuntimeOrigin::signed(5), 1));

            assert_noop!(
                Subject::submit_value(RuntimeOrigin::signed(5), 1),
                Error::<Test>::AlreadySubmitted
            );

            assert_ok!(Subject::reveal(RuntimeOrigin::signed(2), 1, 2));

            assert_noop!(
                Subject::reveal(RuntimeOrigin::signed(2), 1, 2),
                Error::<Test>::AlreadyRevealed
            );

            assert_noop!(
                Subject::reveal(RuntimeOrigin::signed(4), 1, 2),
                Error::<Test>::NotSubmitted
            );

            assert_eq!(Balances::reserved_balance(2), 9);
            assert_eq!(Balances::free_balance(2), 1);

            assert_ok!(Subject::reveal(RuntimeOrigin::signed(3), 3, 4));

            assert_eq!(Balances::reserved_balance(3), 5);
            assert_eq!(Balances::free_balance(3), 1);

            assert_eq!(System::events().len(), 2);

            assert_ok!(Subject::generate_random_number(RuntimeOrigin::signed(1)));

            assert_eq!(System::events().len(), 6);
            assert_eq!(validate_hashes(), 1);

            assert_eq!(Balances::reserved_balance(2), 5);
            assert_eq!(Balances::free_balance(2), 5);

            assert_eq!(Balances::reserved_balance(3), 0);
            assert_eq!(Balances::free_balance(3), 2);

            //

            assert_noop!(
                Subject::submit_hash(RuntimeOrigin::signed(3), hex::encode(new_hash(1, 2))),
                sp_runtime::DispatchError::ConsumerRemaining
            );

            next_block();
            next_minutes();

            assert_ok!(Subject::submit_value(RuntimeOrigin::signed(5), 1));

            assert_ok!(Subject::generate_random_number(RuntimeOrigin::signed(1)));

            assert_eq!(System::events().len(), 7);
            assert_eq!(validate_hashes(), 2);

            next_block();
            next_minutes();

            assert_noop!(
                Subject::generate_random_number(RuntimeOrigin::signed(1)),
                Error::<Test>::RoundNotFinished
            );

            assert_ok!(Subject::submit_value(RuntimeOrigin::signed(5), 2));

            assert_ok!(Subject::generate_random_number(RuntimeOrigin::signed(1)));

            assert_eq!(System::events().len(), 8);
            assert_eq!(validate_hashes(), 3);

            let events = System::events();
            let hashes = events
                .iter()
                .filter_map(|r| match &r.event {
                    RuntimeEvent::Subject(Event::NumberGenerated { hash }) => Some(hash),
                    _ => None,
                })
                .collect::<Vec<&String>>();
            assert_eq!(hashes[0], hashes[1]);
            assert_ne!(hashes[0], hashes[2]);
        });
    }
}
