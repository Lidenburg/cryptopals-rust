#![allow(dead_code)]

use std::collections::HashMap;
use std::fs::File;
use std::io::{Read};
use openssl::symm::{decrypt, encrypt, Cipher, Crypter, Mode};
#[path = "sha1/mod.rs"]
mod sha1;
#[path = "md4/mod.rs"]
mod md4;


pub fn hex_str_to_bytes(in_str: &String) -> Vec<u8>{
	let mut res: Vec<u8> = Vec::new();

	// Jesus christ this is ugly. but it works!
	for (a, b) in in_str.chars().zip(in_str[1..].chars()).step_by(2) {
            // a will have the value 0x31 ('1'). want to turn this into the value 1.
            //println!("a: {:x?}\tb: {:x?}", a, b);

            let val: u8 = (a.to_digit(16).expect("tried to decode a non-hex char") as u8) << 4
                    |
                b.to_digit(16).expect("tried to decode a non-hex char") as u8;

            res.push(val);
	}
	
	return res;
}

/// https://illegalargumentexception.blogspot.com/2015/05/rust-byte-array-to-hex-string.html
pub fn bytes_to_hex_str(bytes: Vec<u8>) -> String {
    let res: Vec<String> = bytes.iter().map(|b| format!("{:02x}", b)).collect();

    return res.join("");
}

/// Takes in a byte array and returns it as a base64 string
pub fn b64_encode(in_str: &[u8]) -> String{
    let alphanum = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/".as_bytes();
    //let mut res: Vec<u8> = Vec::new();
    let mut res: Vec<u8> = Vec::with_capacity(in_str.len());	// Should be faster
    let mut i = 0;

    if in_str.len() == 0 {
        panic!("Tried to base64 encode a 0-length string");
    }

    loop {
        // this shouldnt be possible
        if i >= in_str.len() {
            panic!("At top of base64 loop and no more bytes to use! (Shouldn't be possible)");
        }

        // All but the 2 lowest bits. Want the value as if they were the lowest ones
        res.push(alphanum[((in_str[i + 0] & 0xfc) >> 2) as usize]);

        if i + 1 >= in_str.len() {
            res.push(alphanum[(((in_str[i + 0] & 3) << 4) | 0) as usize]);
            res.push('=' as u8);
            res.push('=' as u8);
            break;
        }

        // 2 Lowest bits from the 1st byte, and the 4 highest from the 2nd.
        res.push(alphanum[(((in_str[i + 0] & 3) << 4) | (in_str[i + 1] & 0xf0) >> 4) as usize]);

        if i + 2 >= in_str.len() {
            res.push(alphanum[(((in_str[i + 1] & 0xf) << 2) | 0) as usize]);
            res.push('=' as u8);
            break;
        }

        // 4 Lowest bits from the 2nd byte, and the 2 highest from the 3rd.
        res.push(alphanum[(((in_str[i + 1] & 0xf) << 2) | (in_str[i + 2] & 0xc0) >> 6) as usize]);

        // 6 lowest bits from the 3rd byte
        res.push(alphanum[(in_str[i + 2] & 0x3f) as usize]);

        i += 3;

        if i >= in_str.len() {
            break;
        }

    }

    //println!("as a string: {:x?}", String::from_utf8_lossy(res.as_slice()));
    return String::from_utf8_lossy(&res).into_owned();
}

/// Takes in a String and returns the raw bytes as a Vec<u8>
pub fn b64_decode(in_str: &str) -> Vec<u8>{
    let mut resvec = Vec::new();
    let mut tmpvec = Vec::new();
    let alphanum = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/".chars();
    let mut mapping = HashMap::new();

    if in_str.len() % 4 != 0 {
        panic!("Base64 string has invalid length!");
    }

    // Slow AF to build this every time but ¯\_(ツ)_/¯
    for (idx, val) in alphanum.enumerate() {
        mapping.insert(val, idx);
    }

    for c in in_str.chars() {
        if mapping.contains_key(&c){
            tmpvec.push(mapping[&c] as u8);
        } else if c == '='{
            break;
        } else {
            panic!("Invalid character in b64_decode!");
        }

    }

    let mut tmp_1 = 0 as u8;
    let mut tmp_2 = 0 as u8;
    let mut tmp_3 = 0 as u8;
    for (idx, val) in tmpvec.iter().enumerate() {
        let calc = idx % 4;
        
        match calc {
            0 => {
                tmp_1 = val << 2;
            },
            1 => {
                // Grab the 2 highest bits (of the 6 bit character). shift them down to the lowest 2 bits
                tmp_1 = tmp_1 | (val & 0x30) >> 4;  
                resvec.push(tmp_1); // this byte is done now

                // the next byte gets its top 4 bits from this chars lowest 4 bits.
                tmp_2 = (val & 0xf) << 4;
            },
            2 => {
                // Grab the 4 highest bits of the char and put them at the lowest 4 bits
                tmp_2 = tmp_2 | (val & 0x3c) >> 2;
                resvec.push(tmp_2);
                // Grab the 2 lowest bits of the char and shift them to the 2 highest bits
                tmp_3 = (val & 0x3) << 6;
            },
            3 => {
                // Grab (all) 6 lowest bits of the char and put them in at the 6 lowest bits
                tmp_3 = tmp_3 | val & 0x3f;

                resvec.push(tmp_3);
            },
            _ => {
                panic!("Not possible!");
            }
        }
    }

    //println!("resvec is: {:?}", resvec);

    return resvec;
}

pub fn fixed_xor(b1: &[u8], b2: &[u8]) -> Vec<u8> {
    let mut vec: Vec<u8> = Vec::with_capacity(b1.len());

    for (a, b) in b1.iter().zip(b2){
        vec.push(a ^ b);
    }

    return vec;
}

pub fn single_key_xor(key: u8, bytes: &[u8]) -> Vec<u8> {
    let mut resves = Vec::with_capacity(bytes.len());
    for b in bytes {
        resves.push(b ^ key);
    }

    return resves;
}

pub fn repeating_key_xor(key: &[u8], to_encrypt: &[u8]) -> Vec<u8> {
    let mut vecout = Vec::with_capacity(to_encrypt.len());

    for i in 0..to_encrypt.len() {
        vecout.push(to_encrypt[i] ^ key[i % key.len()]);
    }

    return vecout;
}

pub fn eng_histogram(eng_str: String) -> f64 {
    let mut score: f64 = 0.0;
    let freqs = vec![0.0651738, 0.0124248, 0.0217339, 0.0349835, 0.1041442, 0.0197881,
        0.0158610, 0.0492888, 0.0558094, 0.0009033, 0.0050529, 0.0331490, 0.0202124, 0.0564513,
        0.0596302, 0.0137645, 0.0008606, 0.0497563, 0.0515760, 0.0729357, 0.0225134, 0.0082903,
        0.0171272, 0.0013692, 0.0145984, 0.0007836, 0.1918182];
    let mut scores = HashMap::new();
    let letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ ".to_string();
    let cleanedup = eng_str.to_uppercase();

    assert!(freqs.len() == letters.len());

    // Building of the frequency values
    for i in 0..freqs.len() {
        scores.insert(letters.as_bytes()[i] as char, freqs[i]);
    }

    for c in cleanedup.chars() {
        if scores.contains_key(&c){
            score += scores[&c];
        }
    }

    return score;
}

/// Returns a tuple where .0 is the score, .1 is the string and .2 is the key
pub fn detect_single_key_xor(enc_str: &Vec<u8>) -> (f64, String, u8){
    let mut bad_chars = Vec::new();
    let mut best_result = (0.0, "".to_string(), 0 as u8);

    for i in 0..31 as u8 {
        bad_chars.push(i);
    }
    for i in 0x7f..0xff as u8 {
        bad_chars.push(i);
    }

    // Newlines are not bad
    bad_chars.remove(10);
    // Tabs are not bad
    bad_chars.remove(09);
    // Carriage returns are not bad
    bad_chars.remove(0x0d);

    for key_try in 0..=255 as u8 {
        let decr = single_key_xor(key_try, enc_str);
        let mut found_bad = false;

        let decr_str = match std::str::from_utf8(&decr) {
            Ok(_v) => _v,
            Err(_e) => {
                //println!("Got an error in from_utf8, continuing");
                continue;
            }
        };

        for bad in bad_chars.iter() {
            if decr_str.as_bytes().contains(bad){
                //println!("Found bad character {}", bad);
                found_bad = true;
                break;
            }
        }
        // Gotta be a better way to do this...
        if found_bad {
            continue;
        }

        let score = eng_histogram(decr_str.to_string());

        if score > best_result.0 {
            best_result.0 = score;
            best_result.1 = decr_str.to_string();  // into owned smthnsmtn?
            best_result.2 = key_try;
        }
    }

    //println!("Returning with score {}", best_result.0);
    return best_result;
}

pub fn hamming_distance(s1: &[u8], s2: &[u8]) -> u32 {
    let mut res: u32 = 0;

    for zipped in s1.iter().zip(s2) {
        let mut diff = zipped.0 ^ zipped.1;

        while diff != 0 {
            res += (diff & 1) as u32;
            diff = diff >> 1;
        }
    }

    return res;
}

pub fn read_in_entire_file(filename: &str) -> Vec<u8> {
    let mut f = File::open(filename).expect(&format!("Failed opening file {}", filename));
    let mut buffer = Vec::new();

    f.read_to_end(&mut buffer).expect("Failed reading to end of file in read_in_entire_file");

    return buffer;
}

pub fn aes_128_ecb_encrypt(plaintext: &[u8], key: &[u8]) -> Vec<u8>{
    //let key = "YELLOW SUBMARINE";

    let cipher = Cipher::aes_128_ecb();
    let encr_res = encrypt(cipher, key, None, &plaintext).expect("Encryption failed!");

    return encr_res;
}

pub fn aes_128_ecb_decrypt(ciphertext: &[u8], key: &[u8]) -> Vec<u8>{
    //let key = "YELLOW SUBMARINE";

    let cipher = Cipher::aes_128_ecb();
    let decr_res = decrypt(cipher, key, None, &ciphertext).expect("ECB Decryption failed!");

    return decr_res;
}

pub fn pkcs7_pad(mut message: Vec<u8>, pad_to: usize) -> Vec<u8>{
    let pad_len = if message.len() % pad_to == 0 {
        pad_to
    } else {
        pad_to - message.len() % pad_to
    };

    //println!("message with length {} getting padded with {:#02x}", message.len(), pad_len);

    for _ in 0..pad_len {
        message.push(pad_len as u8);
    }

    return message;
}

pub fn pkcs7_unpad(mut message: Vec<u8>) -> Result<Vec<u8>, String> {
    let last_byte = *message.last().expect("Why you giving an empty string to unpad?");

    if last_byte == 0 {
        return Err("invalid padding".to_string());

    }

    if message.len() <= last_byte as usize {
        return Err("invalid padding".to_string());
    }

    for i in (message.len() - last_byte as usize..message.len()).rev() {
        if message[i] == last_byte {
            message.pop();
        } else {
            return Err("invalid padding".to_string());
        }
    }

    Ok(message)
}

pub fn aes_128_cbc_encrypt(plaintext: &[u8], key: &[u8], iv: &[u8]) -> Vec<u8>{
    //let mut cipher_xor = iv;
    let mut cipher_xor = vec![0; 16];
    let mut result = Vec::with_capacity(plaintext.len());   // Fast at allocation time and minimal reallocation
    cipher_xor.copy_from_slice(iv);

    assert!(key.len() == 16);
    assert!(iv.len() == 16);

    let plaintext = pkcs7_pad(plaintext.to_vec(), 16);

    for chunk in plaintext.chunks(16){
        //println!("chunk:\t\t\t{:x?}", chunk);

        //println!("chunk after padding:\t{:x?}", chunk);
        let xored = repeating_key_xor(&chunk, &cipher_xor);

        //println!("Encrypting({}) {:x?}", xored.len(), &xored);
        let mut encrypted = aes_128_ecb_encrypt(&xored, key);
        
        // openssl adds a full block of only padding, dont want that
        cipher_xor.copy_from_slice(&encrypted[0..16]);  

        //println!("cbc encrypted({}): {:x?}", cipher_xor.len(), &cipher_xor);
        result.append(&mut encrypted[0..16].to_vec()); // would like an append that doesn't consume...

    }

    assert!(result.len() % 16 == 0);

    //println!("result of encryption: {:x?}", result);
    return result;
}

pub fn aes_128_cbc_decrypt(ciphertext: &[u8], key: &[u8], iv: &[u8]) -> Result<Vec<u8>, String>{
    let mut cipher_xor = vec![0; 16];
    let mut result = Vec::with_capacity(ciphertext.len());   // Fast at allocation time and minimal reallocation

    assert!(key.len() == 16);
    assert!(iv.len() == 16);

    cipher_xor.copy_from_slice(iv); // Should be the IV first time only

    for chunk in ciphertext.chunks(16){
        let mut decrypter = Crypter::new(
            Cipher::aes_128_ecb(),
            Mode::Decrypt,
            key,
            None
            ).expect("failed creating decrypter for aes 128 ecb");

        let mut temp_out = vec![1; 16 * 2];
        decrypter.update(chunk, &mut temp_out).expect("cbc decrypter.update failed");
        //println!("decrypted: {:x?}", temp_out);
        temp_out.truncate(16);
        //println!("decrypted truncated: {:x?}", temp_out);

        let mut plaintext = repeating_key_xor(&cipher_xor, &temp_out);
        //println!("xored: {:x?}", plaintext);
        //println!("xored: {}", std::str::from_utf8(&plaintext).expect("failed decoding decrypted str"));

        result.append(&mut plaintext);

        cipher_xor = chunk.to_vec();
    }

    assert!(result.len() % 16 == 0);

    //println!("cbc_decrypt last byte before unpad: {:x}", result.last().unwrap());
    let result = pkcs7_unpad(result);

    return result;
}

pub fn aes_128_cbc_decrypt_no_unpad(ciphertext: &[u8], key: &[u8], iv: &[u8]) -> Vec<u8> {
    let mut cipher_xor = vec![0; 16];
    let mut result = Vec::with_capacity(ciphertext.len());   // Fast at allocation time and minimal reallocation

    assert!(key.len() == 16);
    assert!(iv.len() == 16);

    cipher_xor.copy_from_slice(iv); // Should be the IV first time only

    for chunk in ciphertext.chunks(16){
        let mut decrypter = Crypter::new(
            Cipher::aes_128_ecb(),
            Mode::Decrypt,
            key,
            None
            ).expect("failed creating decrypter for aes 128 ecb");

        let mut temp_out = vec![1; 16 * 2];
        decrypter.update(chunk, &mut temp_out).expect("cbc decrypter.update failed");
        //println!("decrypted: {:x?}", temp_out);
        temp_out.truncate(16);
        //println!("decrypted truncated: {:x?}", temp_out);

        let mut plaintext = repeating_key_xor(&cipher_xor, &temp_out);
        //println!("xored: {:x?}", plaintext);
        //println!("xored: {}", std::str::from_utf8(&plaintext).expect("failed decoding decrypted str"));

        result.append(&mut plaintext);

        cipher_xor = chunk.to_vec();
    }

    assert!(result.len() % 16 == 0);

    //println!("cbc_decrypt last byte before unpad: {:x}", result.last().unwrap());
    //let result = pkcs7_unpad(result);

    return result;
}

pub fn aes_128_ctr_encrypt(plaintext: &[u8], key: &[u8], nonce: u64) -> Vec<u8> {
    let mut counter = 0u64;
    let mut encrypted = Vec::with_capacity(plaintext.len());

    for chunk in plaintext.chunks(16) {
        let mut to_encr = nonce.to_le_bytes().to_vec();
        to_encr.extend(&counter.to_le_bytes());

        let encd = aes_128_ecb_encrypt(&to_encr, key);

        for (idx, b) in chunk.iter().enumerate() {
            encrypted.push(b ^ encd[idx]);
        }

        counter += 1;
    }

    return encrypted;
}

pub fn aes_128_ctr_crypt_with_start_counter(plaintext: &[u8], key: &[u8], nonce: u64, counter_start: u64) -> Vec<u8> {
    let mut counter = counter_start;
    let mut encrypted = Vec::with_capacity(plaintext.len());

    for chunk in plaintext.chunks(16) {
        let mut to_encr = nonce.to_le_bytes().to_vec();
        to_encr.extend(&counter.to_le_bytes());

        let encd = aes_128_ecb_encrypt(&to_encr, key);

        for (idx, b) in chunk.iter().enumerate() {
            encrypted.push(b ^ encd[idx]);
        }

        counter += 1;
    }

    return encrypted;
}

pub fn aes_128_ctr_decrypt(plaintext: &[u8], key: &[u8], nonce: u64) -> Vec<u8> {
    return aes_128_ctr_encrypt(plaintext, key, nonce);
}

pub struct Rand {
    val: u64,
}

impl Rand {
    pub fn new(seed: u64) -> Rand {
        //println!("Making a new rand object with seed: {}", seed);
        return Rand {val: seed};
    }

    pub fn srand(&mut self, seed: u64) {
        self.val = seed;
    }

    /// Updates the internal state
    fn xorshift(&mut self){
        let mut x = self.val ^ (self.val << 13);
        x ^= x >> 17;
        x ^= x << 5;
        self.val = x;
    }

    pub fn rand_u64(&mut self) -> u64 {
        self.xorshift();
        return self.val;
    }

    pub fn rand_u32(&mut self) -> u32 {
        self.xorshift();
        return self.val as u32;
    }

    pub fn rand_u8(&mut self) -> u8 {
        self.xorshift();
        return self.val as u8;
    }

    pub fn rand_u8_vec(&mut self, size: usize) -> Vec<u8>{
        let mut result = Vec::with_capacity(size);
        //let mut rand = Rand::new(133711337);

        // TODO: implement xor-shift
        for _ in 0..size {
            result.push(self.rand_u8());
        }

        return result;
    }

}

// Maybe move this to inside of the rand class?
// Assume 32-bit mt. not specified though

pub struct RandMT {
    mt: [u32; RandMT::N],
    index: usize,
}

impl RandMT {
    pub const W: u32 = 32;
    pub const N: usize = 624;
    pub const M: usize = 397;
    //pub const R: usize = 31;  // Never used
    pub const A: u32 = 0x9908B0DF;
    pub const U: usize = 11;
    //pub const D: u32 = 0xFFFFFFFF;    // Never used
    pub const S: usize = 7;
    pub const B: u32 = 0x9D2C5680;
    pub const T: usize = 15;
    pub const C: u32 = 0xEFC60000;
    pub const L: usize = 18;
    pub const F: u32 = 1812433253;
    pub const LOWER_MASK: u32 = 0x7FFFFFFF;
    pub const UPPER_MASK: u32 = 0x80000000;

    pub fn seed(&mut self, seed_val: u32) {
        self.index = RandMT::N;
        self.mt[0] = seed_val;

        for i in 1..=RandMT::N - 1{
            // MT[i] := lowest w bits of (f * (MT[i-1] xor (MT[i-1] >> (w-2))) + i)
            self.mt[i] = RandMT::F.overflowing_mul(self.mt[i - 1] ^ (self.mt[i - 1] >> (RandMT::W - 2))).0.overflowing_add(i as u32).0;
        }
    }

    pub fn rand_u32(&mut self) -> u32 {
        if self.index == RandMT::N {
            self.twist();
        } else if self.index > RandMT::N {
            panic!("Generator was never seeded! (Shouldn't even be possible)");
        }

        let mut y: u32 = self.mt[self.index];

        y ^= y >> RandMT::U;
        y ^= (y << RandMT::S) & RandMT::B;
        y ^= (y << RandMT::T) & RandMT::C;
        y ^= y >> RandMT::L;

        self.index += 1;

        return y;
    }

    pub fn rand_u8(&mut self) -> u8 {
        return (self.rand_u32() % 256) as u8;
    }

    pub fn rand_u8_vec(&mut self, size: usize) -> Vec<u8> {
        let mut res = Vec::new();

        for _ in 0..size {
            res.push(self.rand_u8());
        }

        return res;
    }

    pub fn twist(&mut self) {
        for i in 0..=RandMT::N - 1 {
            let x = (self.mt[i] & RandMT::UPPER_MASK).overflowing_add(
                self.mt[(i + 1) % RandMT::N] & RandMT::LOWER_MASK).0;

            let mut xa = x >> 1;

            if x % 2 != 0 {
                xa ^= RandMT::A;
            }

            self.mt[i] = self.mt[(i + RandMT::M) % RandMT::N] ^ xa;
        }

        self.index = 0;
    }

    pub fn new(seed: u32) -> RandMT {
        let mut rng = RandMT {mt: [0; RandMT::N], index: 0};

        rng.seed(seed);

        rng
    }

    pub fn from_state(state: &[u32]) -> RandMT {
        let mut l_mt = [0u32; RandMT::N];
        l_mt.copy_from_slice(state);
        let rng = RandMT {mt: l_mt, index: RandMT::N};

        return rng;
    }

    pub fn stream_cipher_crypt(plaintext: &[u8], seed: u16) -> Vec<u8> {
        let mut rng = RandMT::new(seed as u32);
        let plainlen = plaintext.len();
        let keystream = rng.rand_u8_vec(plainlen);
        let mut encrypted = Vec::with_capacity(plainlen);

        for i in 0..plainlen {
            encrypted.push(keystream[i] ^ plaintext[i]);
        }

        return encrypted;
    }
}

fn untemper_left_and(val: u32, shift_val: u32, magic_and: u32) -> u32 {
    let loop_count = (32 / shift_val) + ((32 % shift_val != 0) as u32);
    let mut res_bits = Vec::with_capacity(loop_count as usize);
    let mut mask = 2u32.pow(shift_val) - 1;

    res_bits.push(val & mask);

    for i in 1..loop_count {
        mask = mask << shift_val;
        res_bits.push((val & mask) ^ ((res_bits[i as usize - 1] << shift_val) & mask & magic_and));
    }

    let mut as_u32 = 0;
    for bits in res_bits {
        as_u32 |= bits;
    }

    //println!("as_u32: {}", as_u32);
    return as_u32;
}

fn untemper_right(val: u32, shift_val: u32) -> u32 {
    let loop_count = (32 / shift_val) + ((32 % shift_val != 0) as u32);
    let mut res_bits = Vec::with_capacity(loop_count as usize);
    let mut mask = 2u32.pow(shift_val) - 1;
    mask = mask.rotate_right(shift_val);

    res_bits.push(val & mask);

    for i in 1..loop_count {
        mask = mask >> shift_val;
        res_bits.push((val & mask) ^ (res_bits[i as usize - 1] >> shift_val));
    }

    let mut as_u32 = 0;
    for bits in res_bits {
        as_u32 |= bits;
    }

    return as_u32;
}

// https://krypt05.blogspot.com/2015/10/reversing-shift-xor-operation.html
pub fn mt_untemper(val: u32) -> u32 {
    let y3 = untemper_right(val, 18);
    let y2 = untemper_left_and(y3, 15, 0xEFC60000);
    let y1 = untemper_left_and(y2, 7, 0x9D2C5680);
    let y0 = untemper_right(y1, 11);

    //println!("final result: {:x}", y0);

    return y0;
    
}

pub fn sha1sum(to_sum: &[u8]) -> [u8; 20] {
    sha1::Sha1::from(&to_sum).digest().bytes()
}

/// Returns the padding bytes that will be used in sha1
pub fn sha1_padding_for(data: &[u8]) -> Vec<u8> {
    sha1::Sha1::get_padding_bytes(data)
}

/// Returns the state from a SHA1 output
pub fn sha1_state_from(hash: [u8; 20]) -> [u32; 5] {
    let mut tmparr = [0u8; 4];

    tmparr.clone_from_slice(&hash[0..4]);
    let val1 = u32::from_be_bytes(tmparr);

    tmparr.clone_from_slice(&hash[4..8]);
    let val2 = u32::from_be_bytes(tmparr);

    tmparr.clone_from_slice(&hash[8..12]);
    let val3 = u32::from_be_bytes(tmparr);

    tmparr.clone_from_slice(&hash[12..16]);
    let val4 = u32::from_be_bytes(tmparr);

    tmparr.clone_from_slice(&hash[16..20]);
    let val5 = u32::from_be_bytes(tmparr);

    [val1, val2, val3, val4, val5]
    //sha1::Sha1::from(&data).digest().u32s()
}

/// Start pos should always be divisible by 64. no?
pub fn sha1_append(state: [u32; 5], append_data: &[u8], start_pos: usize) -> [u8; 20] {
    let mut hasher = sha1::Sha1::new();
    hasher.update("A".repeat(start_pos).as_bytes());
    // sha object is now in a good internal state.

    // Proceed with our own state
    hasher.update_with_state(append_data, state);

    return hasher.digest().bytes();
}

pub fn sha1_from_state(state: [u32; 5], data: &[u8], len: u64) -> [u8; 20] {
    let mut hasher = sha1::Sha1::new_with_state(state, len);
    hasher.update(data);

    return hasher.digest().bytes();
}

pub fn create_sha1_mac(secret: &[u8], message: &[u8]) -> [u8; 20] {
    let mut combined = vec![0; secret.len()];
    combined.clone_from_slice(secret);
    combined.extend(message);

    let digest = sha1sum(&combined);

    return digest;
}

pub fn verify_sha1_mac(mac: &[u8; 20], secret: &[u8], message: &[u8]) -> bool {
    return mac == &create_sha1_mac(secret, message)[..];
}

pub fn md4sum(data: &[u8]) -> [u8; 16] {
    let sum = md4::md4(data.to_vec());

    let [b00, b01, b02, b03] = sum[0].to_be_bytes();
    let [b10, b11, b12, b13] = sum[1].to_be_bytes();
    let [b20, b21, b22, b23] = sum[2].to_be_bytes();
    let [b30, b31, b32, b33] = sum[3].to_be_bytes();

    [b00, b01, b02, b03, b10, b11, b12, b13, b20, b21, b22, b23, b30, b31, b32, b33]
}

/// Returns the padding bytes that will be used in md4
pub fn md4_padding_for(data: &[u8]) -> Vec<u8> {
    md4::get_padding_bytes(data.to_vec())
}

pub fn create_md4_mac(secret: &[u8], message: &[u8]) -> [u8; 16] {
    let mut combined = vec![0; secret.len()];
    combined.clone_from_slice(secret);
    combined.extend(message);

    let digest = md4sum(&combined);

    return digest;
}

pub fn verify_md4_mac(mac: &[u8; 16], secret: &[u8], message: &[u8]) -> bool {
    return mac == &create_md4_mac(secret, message)[..];
}

/// Returns the state from a MD4 output
pub fn md4_state_from(hash: [u8; 16]) -> [u32; 4] {
    let mut tmparr = [0u8; 4];

    tmparr.clone_from_slice(&hash[0..4]);
    let val1 = u32::from_le_bytes(tmparr);

    tmparr.clone_from_slice(&hash[4..8]);
    let val2 = u32::from_le_bytes(tmparr);

    tmparr.clone_from_slice(&hash[8..12]);
    let val3 = u32::from_le_bytes(tmparr);

    tmparr.clone_from_slice(&hash[12..16]);
    let val4 = u32::from_le_bytes(tmparr);

    [val1, val2, val3, val4]
    //sha1::Sha1::from(&data).digest().u32s()
}

pub fn md4_from_state(state: [u32; 4], data: &[u8], len: usize) -> [u8; 16] {
    let sum = md4::md4_with_state(data, state, len);

    // Orig
    let [b00, b01, b02, b03] = sum[0].to_be_bytes();
    let [b10, b11, b12, b13] = sum[1].to_be_bytes();
    let [b20, b21, b22, b23] = sum[2].to_be_bytes();
    let [b30, b31, b32, b33] = sum[3].to_be_bytes();


    [b00, b01, b02, b03, b10, b11, b12, b13, b20, b21, b22, b23, b30, b31, b32, b33]
}

fn hmac_sha1_pad(data: &[u8]) -> [u8; 64] {
    let mut res = [0u8; 64];
    assert!(data.len() <= 64);

    for i in 0..data.len() {
        res[i] = data[i];
    }
    for i in data.len()..64 {
        res[i] = 0;
    }

    return res;
}

pub fn create_sha1_hmac(secret: &[u8], message: &[u8]) -> [u8; 20] {
    let secret = if secret.len() > 64 {
        sha1sum(secret).to_vec()
    } else {
        secret.to_vec()
    };

    let secret = if secret.len() < 64 {
        hmac_sha1_pad(&secret).to_vec()
    } else {
        secret.to_vec()
    };

    let o_key_pad = repeating_key_xor(&secret, "\x5c".repeat(64).as_bytes());
    let i_key_pad = repeating_key_xor(&secret, "\x36".repeat(64).as_bytes());

    let mut inner = Vec::new();
    inner.extend(&i_key_pad);
    inner.extend(message);
    let inner_hash = sha1sum(&inner);

    let mut outer = Vec::new();
    outer.extend(&o_key_pad);
    outer.extend(&inner_hash);

    let outer_hash = sha1sum(&outer);

    outer_hash
}

pub fn verify_sha1_hmac(hmac: &[u8; 20], secret: &[u8], message: &[u8]) -> bool {
    hmac == &create_sha1_hmac(secret, message)
}
