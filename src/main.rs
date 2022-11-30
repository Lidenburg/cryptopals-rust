#![allow(non_snake_case)]

mod helpers;

#[derive(Debug)]
pub struct Profile {
    email: String,
    uid: u32,
    role: String,
}

mod kv_parse {
    use super::Profile;
    pub fn deserialize(serialized: &str) -> Option<Profile> {
        let mut serialized = serialized.clone();
        let mut kvs = Vec::new();
        let mut prof: Profile = Profile{email: "".to_string(), uid: 0, role: "".to_string()};

        loop {
            let start = serialized.find("&");

            let start = match start {
                Some(val) => val,
                None => {
                    kvs.push(&serialized[..]);
                    break;
                },
            };

            kvs.push(&serialized[..start]);
            serialized = &serialized[start+1..];
        }
        //println!("key value pairs: {:?}", kvs);

        for item in kvs {
            //println!("item: {}", item);
            let name_offset = item.find("=");

            let name_offset = if let Some(val) = name_offset {
                val
            } else {
                panic!("AAAAHHH!!");
            };

            let name = &item[..name_offset];
            //println!("name: {}", &name);

            let value = &item[name_offset + 1..];

            match name {
                "email" => {
                    if let Some(_) = value.find("="){
                        ()
                    } else {
                        prof.email = value.to_string()
                    }
                },
                "uid" => prof.uid = value.parse::<u32>().unwrap(),
                "role" => prof.role = value.to_string(),
                _ => panic!("invalid key name!"),
            }
            //println!("value: {}\n", &item[name_offset + 1..]);
        }

        if prof.email == "" || prof.uid == 0 || prof.role == "" {
            return None;
        }

        //println!("profile: {:#?}", prof);
        return Some(prof);
    }

    pub fn serialize(prof: &Profile) -> String {
        // No iterators for structs :(. Actually this is way more nice than an iterator.
        format!("email={}&uid={}&role={}", prof.email, prof.uid, prof.role)
    }

    /// Returns the serialized form of the Profile for the email
    pub fn profile_for(email: &str) -> String {
        let prof = Profile {
            email: email.to_string(),
            uid: 10,
            role: "user".to_string(),
        };

        return serialize(&prof);
    }

    pub fn sanity_checks() {
        // Some sanity checks
        match deserialize("email=foo@bar.com&uid=10&role=user") {
            //Some(_) => println!("correctly serialized object succeeded"),
            Some(_) => (),
            None => panic!("correctly formatted serialized object failed!"),
        }
        match deserialize("email=f=o@bar.com&uid=10&role=user") {
            //Some(_) => panic!("incorrectly formatted serialized object succeeded!"),
            Some(_) => panic!("incorrectly formatted serialized object succeeded!"),
            None => (),
        }
        let serialized = "email=foo@bar.com&uid=10&role=user";
        let prof = deserialize(serialized);
        let prof = match prof {
            Some(val) => val,
            None => panic!("failed deserializing"),
        };
        let round_trip = serialize(&prof);
        //println!("round_trip: {}", round_trip);
        assert!(serialized == &round_trip);
    }
}

mod set1 {
    use std::fs::File;
    use std::io::{BufRead, BufReader};
    use super::helpers;

    pub fn challenge_1(){
        let input = "49276d206b696c6c696e6720796f757220627261696e206c696b65206120706f69736f6e6f7573206d757368726f6f6d".to_string();
        let correct_output = "SSdtIGtpbGxpbmcgeW91ciBicmFpbiBsaWtlIGEgcG9pc29ub3VzIG11c2hyb29t".to_string();
        let res = helpers::b64_encode(helpers::hex_str_to_bytes(&input).as_slice());
    
        assert!(res == correct_output);
        println!("Challenge 1: Successful!");
        
        //println!("Hex decode of \"{}\": {:x?}", to_decode, b64_encode(hex_str_to_bytes(to_decode).as_slice()));
    }
    
    pub fn challenge_2(){
        let input = "1c0111001f010100061a024b53535009181c".to_string();
        let correct_output = "746865206b696420646f6e277420706c6179".to_string();
        let res_vec = helpers::fixed_xor(&helpers::hex_str_to_bytes(&input), &helpers::hex_str_to_bytes(&"686974207468652062756c6c277320657965".to_string()));
        let res = helpers::bytes_to_hex_str(res_vec);
    
        assert!(res == correct_output);
        println!("Challenge 2: Successful!");
    }
    
    pub fn challenge_3(){
        let input = "1b37373331363f78151b7f2b783431333d78397828372d363c78373e783a393b3736".to_string();
    
        let res = helpers::detect_single_key_xor(&helpers::hex_str_to_bytes(&input));
    
        assert!(res.1 == "Cooking MC's like a pound of bacon".to_string());
        println!("Challenge 3: Successful! Found best match to be: \"{}\" with score {}", res.1, res.0);
    
    }
    
    pub fn challenge_4() {
        let file = File::open("4.txt").expect("Failed opening 4.txt");
        let reader = BufReader::new(file);
        let mut score = 0.0;
        let mut best_str = "".to_string();
        
        // Read file line by line
        for (_, l) in reader.lines().enumerate() {
            let res = helpers::detect_single_key_xor(&helpers::hex_str_to_bytes(&l.unwrap()));
    
            if res.0 > score {
                best_str = res.1;
                score = res.0;
            }
        }
    
        assert!(best_str == "Now that the party is jumping\n".to_string());
    
        println!("Challenge 4: Successful! Found best match to be: \"{}\" with score {}",
            best_str.replace("\n", ""), score);
        //println!("Challenge 4: Successful! Best match was \"{}\" with score {}", best_str, score);
    }
    
    pub fn challenge_5() {
        let plain = "Burning 'em, if you ain't quick and nimble\nI go crazy when I hear a cymbal".to_string();
        let key = "ICE".to_string();
    
        let res = helpers::bytes_to_hex_str(helpers::repeating_key_xor(key.as_bytes(), plain.as_bytes()));
    
        assert!(res == "0b3637272a2b2e63622c2e69692a23693a2a3c6324202d623d63343c2a26226324272765272a282b2f20430a652e2c652a3124333a653e2b2027630c692b20283165286326302e27282f".to_string());
    
        println!("Challenge 5: Successful!");
    
    }
    
    pub fn challenge_6(){
        assert!(
            helpers::hamming_distance("this is a test".as_bytes(), "wokka wokka!!!".as_bytes())
            ==
            37);
    
        let tmp = helpers::read_in_entire_file("6.txt");
        let file_contents = std::str::from_utf8(&tmp).unwrap().replace("\n", "");
        let raw_file_contents = helpers::b64_decode(&file_contents.to_string());
        let mut best_distance = 999999999.0;
        let mut keysize_guess = 0xdeadbeef;
    
        for keysize in 2..40 {
            let mut distance = 0.0;
            for ham in 1..10 {
                // Not what I first intended to do but hey, it works
                distance += helpers::hamming_distance(&raw_file_contents[0..keysize],
                    &raw_file_contents[keysize*ham..keysize*(ham + 1)]) as f64
                    /
                    keysize as f64;
    
            }
    
            distance /= 10 as f64;
    
            if distance < best_distance {
                //println!("Edit distance with keysize {}: {}", keysize, distance);
                best_distance = distance;
                keysize_guess = keysize;
            }
        }
    
        if keysize_guess == 0xdeadbeef {
            panic!("Didn't find a good keysize guess!");
        } else {
            //println!("Guessing keysize is {}", keysize_guess);
        }
    
        let mut broken_up: Vec<u8> = Vec::with_capacity(raw_file_contents.len() / keysize_guess);
    
        for i in (0..raw_file_contents.len()).step_by(keysize_guess) {
            // Sooo sloow
            if i + keysize_guess >= raw_file_contents.len() {
                //println!("Got last block special case");
                //broken_up.push(&raw_file_contents[i..]);
                broken_up.extend_from_slice(&raw_file_contents[i..]);
            } else {
                //broken_up.push(&raw_file_contents[i..i+keysize_guess]);
                broken_up.extend_from_slice(&raw_file_contents[i..i+keysize_guess]);
            }
        }
    
        let mut transposed: Vec<Vec<u8>> = Vec::with_capacity(keysize_guess);
    
        for _ in 0..keysize_guess {
            let tmp = Vec::new();
            transposed.push(tmp);
        }
    
        for i in 0..broken_up.len() {
            transposed[i % keysize_guess].push(broken_up[i]);
        }
    
        let mut key: Vec<u8> = Vec::with_capacity(transposed.len());
    
        for i in &transposed {
            let res = helpers::detect_single_key_xor(i);
    
            key.push(res.2);
        }
    
        let key = std::str::from_utf8(key.as_slice()).expect("failed converting key to string");
    
        assert!(key == "Terminator X: Bring the noise".to_string());
    
        println!("Challenge 6: Successful! Key was: {}", key);
    }
    
    pub fn challenge_7() {
        let file_contents = helpers::read_in_entire_file("7.txt");
        let file_contents = helpers::b64_decode(&std::str::from_utf8(&file_contents).expect("failed making string from 7.txt").to_string().replace("\n", ""));
    
        let key = "YELLOW SUBMARINE";
    
        //let decr_res = decrypt(cipher, key.as_bytes(), None, &file_contents).expect("Decryption failed!");
        let decr_res = helpers::aes_128_ecb_decrypt(&file_contents, key.as_bytes());
    
        //println!("decr_res: {}", std::str::from_utf8(&decr_res).expect("Failed converting decrypted data to utf8"));
    
        //println!("{:?}", &decr_res[0..33]);
        //println!("{:?}", "I'm back and I'm ringin' the bell".as_bytes());
        assert!(&decr_res[0..33] == "I'm back and I'm ringin' the bell".as_bytes());
    
        println!("Challenge 7: Successful!");
    }
    
    pub fn is_ecb_encrypted(input: Vec<u8>) -> bool{
        let mut chunked = Vec::new();
        // Divide it up into 16 byte (128 bit) chunks
        for i in (0..input.len()).step_by(16){
            chunked.push(&input[i..i+16]);
        }

        for (idx, chunk) in chunked.iter().enumerate() {
            if chunked[idx + 1..].contains(chunk) == true{
                return true;
                //println!("Found duplets! on row {}", row + 1);
            }
        }

        return false;
    }

    pub fn challenge_8(){
        let file = File::open("8.txt").expect("Failed opening 8.txt");
        let reader = BufReader::new(file);
        let mut ecb_row = 0;
    
        // Read file line by line
        for (row, l) in reader.lines().enumerate() {
            let raw = helpers::hex_str_to_bytes(&l.unwrap());
    
            //let mut chunked = Vec::new();
            // Divide it up into 16 byte (128 bit) chunks
            //for i in (0..raw.len()).step_by(16){
            //    chunked.push(&raw[i..i+16]);
            //}
    
            //for (idx, chunk) in chunked.iter().enumerate() {
            //    if chunked[idx + 1..].contains(chunk) == true{
            //        ecb_row = row + 1;
            //        //println!("Found duplets! on row {}", row + 1);
            //    }
            //}

            if is_ecb_encrypted(raw) {
                ecb_row = row + 1;
            }
        }
    
        assert!(ecb_row == 133);
        println!("Challenge 8: Successful! Ecb mode found on row {}", ecb_row);
    }
}

mod set2 {
    use super::helpers;
    use std::time::{SystemTime, UNIX_EPOCH};
    use std::collections::HashMap;
    use std::io::{stdout, Write};
    use super::kv_parse as kv_parse;

    /// PKCS#7 padding
    pub fn challenge_9(){
        let res = helpers::pkcs7_pad("YELLOW SUBMARINE".as_bytes().to_vec(), 20);
        println!("Challenge 9: Successful! Padding was {:x}", res[res.len() - 1]);
    }

    pub fn challenge_10(){
        let key = b"YELLOW SUBMARINE";
        let iv = vec![0; 16];

        let file_contents = helpers::read_in_entire_file("10.txt");
        let file_contents = std::str::from_utf8(&file_contents).expect("failed making 10.txt to string");
        let file_contents = file_contents.replace("\n", "");

        let raw = helpers::b64_decode(&file_contents);

        let res = helpers::aes_128_cbc_decrypt(&raw, key, &iv).unwrap();

        assert!(&res[0..33] == "I'm back and I'm ringin' the bell".as_bytes());

        //println!("challenge 10 decrypted: {}", std::str::from_utf8(&res).unwrap());

        println!("Challenge 10: Successful!");

    }

    enum EncKind {
        ECB,
        CBC,
    }

    pub fn challenge_11(){
        fn enc_oracle(arg_input: &Vec<u8>) -> (Vec<u8>, EncKind){
            //let mut rng = helpers::Rand::new(12341337);
            // JEEEEEssus that is one long line
            let mut rng = helpers::Rand::new(SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_micros() as u64);
            let key = rng.rand_u8_vec(16);
            let iv = rng.rand_u8_vec(16);
            let enc_type = if rng.rand_u8() % 2 == 0 {
                EncKind::ECB
            } else {
                EncKind::CBC
            };
            let mut input = arg_input.clone();

            // append 5-10 bytes at start
            let rand_append = (rng.rand_u8() % 5) + 5;
            //println!("Appending {} random bytes at start", rand_append);
            let rand_bytes = rng.rand_u8_vec(rand_append as usize);
            // splice to insert a slice into a vector
            input.splice(0..0, rand_bytes);

            // append 5-10 bytes at end
            let rand_append = (rng.rand_u8() % 5) + 5;
            //println!("Appending {} random bytes at end", rand_append);
            let mut rand_bytes = rng.rand_u8_vec(rand_append as usize);
            input.append(&mut rand_bytes);

            //println!("input with random: {:x?}", input);

            let output = match enc_type {
                EncKind::ECB => helpers::aes_128_ecb_encrypt(&input, &key),
                EncKind::CBC => helpers::aes_128_cbc_encrypt(&input, &key, &iv),
            };

            //println!("Random encrypted data: {:x?}", output);

            return (output, enc_type);
            }
        //println!("key: {:x?}", key);
        // If we have four block-size blocks of known content, we will have
        // 2 repeating blocks in the case of ECB mode even if the oracle
        // appends random bytes before and after our blocks.
        let needle = "A".repeat(16).repeat(4).as_bytes().to_vec();

        // Test our heuristic by making sure we get it right 100 times w/o failing
        for _ in 0..100 {
            let rand_data = enc_oracle(&needle);

            let mode_guess = if super::set1::is_ecb_encrypted(rand_data.0) {
                EncKind::ECB
            } else {
                EncKind::CBC
            };


            match (mode_guess, rand_data.1) {
                (EncKind::ECB, EncKind::ECB) => (),
                (EncKind::CBC, EncKind::CBC) => (),
                (EncKind::ECB, EncKind::CBC) => panic!("no match!"),
                (EncKind::CBC, EncKind::ECB) => panic!("no match!"),
            };

        }

        println!("Challenge 11: Successful! Detected all ECB/CBC encryptions correctly!");
    }

    pub fn challenge_12(){
        fn enc_oracle(arg_input: &Vec<u8>) -> Vec<u8>{
            //let mut rng = helpers::Rand::new(12341337);
            // JEEEEEssus that is one long line
            let mut rng = helpers::Rand::new(1337); // constant seed to always get the same key
            let key = rng.rand_u8_vec(16);
            let mut input = arg_input.clone();

            // Append this unknown blob at the start
            let mut raw =
                helpers::b64_decode(&"Um9sbGluJyBpbiBteSA1LjAKV2l0aCBteSByYWctdG9wIGRvd24gc28gbXkgaGFpciBjYW4gYmxvdwpUaGUgZ2lybGllcyBvbiBzdGFuZGJ5IHdhdmluZyBqdXN0IHRvIHNheSBoaQpEaWQgeW91IHN0b3A/IE5vLCBJIGp1c3QgZHJvdmUgYnkK".to_string());

            // input = arg_input + secret
            input.append(&mut raw);

            let output = helpers::aes_128_ecb_encrypt(&input, &key);

            return output;
            }

        //println!("Doing Challenge 12");
        print!("Challenge 12: ");
        let _ = stdout().flush();

        let mut res = Vec::new();
        let encrypted = enc_oracle(&b"A".to_vec());
        let encrypted_len = encrypted.len();
        //println!("encrypted({}): {:x?}", encrypted.len(), encrypted);

        // Find how many bytes we need to append to make the length increase.
        // that means we filled the entire block and openssl had to make another one with only
        // padding in it
        let mut req_size = 0;

        for i in 0..16 {
            let encrypted = enc_oracle(&"A".repeat(i).as_bytes().to_vec());
            if encrypted.len() != encrypted_len {
                req_size = i;
                break;
            }
        }

        // Make sure it's really ECB mode
        let needle = "A".repeat(16).repeat(4).as_bytes().to_vec();
        let mode = super::set1::is_ecb_encrypted(enc_oracle(&needle));

        if mode == false {
            panic!("Guessed wrong in challenge 12!");
        }

        //println!("req size: {}", req_size);
        //println!("size of secret: {}", encrypted_len - 16);

        // Start of the actual logic
        // One byte short since we always want a byte from the secret part
        let mut known_chunk = "A".repeat(((encrypted_len - req_size) / 16) * 16 + 16 - 1).to_string().as_bytes().to_vec();
        let mut precomp_block = "A".repeat(16).to_string().as_bytes().to_vec();

        for _ in 0..encrypted_len - req_size {
            let mut lookup = HashMap::new();
            // Build up the mapping for all values
            for c in 1u8..127 {
                // Always want to change the last byte
                let mut arr = [0; 16];
                precomp_block[15] = c;

                let encrypted = enc_oracle(&precomp_block);
                arr.copy_from_slice(&encrypted[0..16]);
                //println!("mapping {:x?} to -> {:x}", &encrypted[0..16], c);
                lookup.insert(arr, c);
            }

            let encrypted = enc_oracle(&known_chunk);
            
            let mut tmparr = [0; 16];
            tmparr.copy_from_slice(&encrypted[encrypted_len - 16 * 1..encrypted_len]);

            let last_byte = lookup.get(&tmparr);

            let last_byte = match last_byte {
                Some(val) => *val,
                None => {
                    println!("What we got so far: {}", std::str::from_utf8(&res).unwrap());
                    panic!("No such entry in the hashmap!")
                },
            };

            res.push(last_byte);

            if known_chunk.len() == 0 {
                break;
            }

            // Remove another byte so that we actually decrease in size and get
            // new bytes from the secret
            known_chunk.remove(0);

            precomp_block[15] = last_byte;
            precomp_block.rotate_left(1);
        }

        let secret_decoded = helpers::b64_decode(&"Um9sbGluJyBpbiBteSA1LjAKV2l0aCBteSByYWctdG9wIGRvd24gc28gbXkgaGFpciBjYW4gYmxvdwpUaGUgZ2lybGllcyBvbiBzdGFuZGJ5IHdhdmluZyBqdXN0IHRvIHNheSBoaQpEaWQgeW91IHN0b3A/IE5vLCBJIGp1c3QgZHJvdmUgYnkK".to_string());

        let resstr = std::str::from_utf8(&res).expect("Definitely something wrong!!");
        assert!(resstr == std::str::from_utf8(&secret_decoded).unwrap());

        //println!("result: {}", resstr);
        println!("Successful! Decrypted with oracle!");

    }

//    fn print_in_chunks(a: &Vec<u8>) {
//        //let mut tmp = Vec::new();
//        for (idx, b) in a.chunks(16).enumerate() {
//            //tmp.push(b);
//            print!("{}: {:x?} ", idx, b);
//        }
//        println!("");
//    }

    pub fn challenge_13(){
        kv_parse::sanity_checks();
        fn enc_prof(email: &str, key: &[u8])-> Vec<u8> {
            helpers::aes_128_ecb_encrypt(kv_parse::profile_for(email).as_bytes(), key)
        }
        fn check_success(encrypted_prof: &[u8], key: &[u8]) -> bool {
            let decrypted = helpers::aes_128_ecb_decrypt(&encrypted_prof, key);
            let decrypted_str = std::str::from_utf8(&decrypted);
            let decrypted_str = match decrypted_str {
                Ok(v) => v,
                Err(_) => return false,
            };
            //println!("check_success deserializing '{}'", decrypted_str);
            let prof = kv_parse::deserialize(decrypted_str);
            let prof = match prof {
                Some(v) => v,
                None => return false,
            };

            if prof.role == "admin".to_string() {
                return true;
            } else {
                return false;
            }
        }
        let mut rng = helpers::Rand::new(1337); // constant seed to always get the same key
        let key = rng.rand_u8_vec(16);
        //let mut req_size = 0;   // Bytes required to fill every block
        //let start_len = enc_prof(&"A".repeat(16 * 5), &key).len();

        // Check that the check_success function works
        {
        let tmp = "email=foo&uid=10&role=admin";
        let encd = helpers::aes_128_ecb_encrypt(tmp.as_bytes(), &key);
        assert!(check_success(&encd, &key) == true);
        }

        // 16 * N A's will result in N - 1 blocks of the same recognizable block
        let enc = enc_prof(&"A".repeat(16 * 5), &key);
        let is_ecb = super::set1::is_ecb_encrypted(enc);
        assert!(is_ecb == true);

        // Figure out how many bytes are needed to create a new block
        // NOTE: hardcoded
        //for i in 0..16 {
        //    if enc_prof(&"A".repeat(16 * 5 + i), &key).len() != start_len {
        //        req_size = i;
        //        break;
        //    }
        //}

        // req_size A's will fill the first block up to 16. the other 2 blocks will serve as a
        // pattern to search for (2 blocks with the same contents since ECB mode)
        let mut known_chunk = "A".repeat(16 * 3 - "email=".len());

        // this will become the 3rd block
        //known_chunk += std::str::from_utf8(&helpers::pkcs7_pad("admin".as_bytes().to_vec(), 16)).unwrap();
        known_chunk += &"admin\x0b\x0b\x0b\x0b\x0b\x0b\x0b\x0b\x0b\x0b\x0b\x0b";
        known_chunk += &"A".repeat(16 - 2); // A bit magic const

        // bytes needed to push 'user' from 'role=user' into its own block with padding appended
        known_chunk += &"A".repeat("user".len());

        let enc = enc_prof(&known_chunk, &key);
        //println!("enc: ");
        //print_in_chunks(&enc);

        let mut last_chunk = vec![0; 16];
        let mut as_chunk_idx = 0;
        for (idx, chunk) in enc.chunks(16).enumerate() {
            if last_chunk == chunk {
                as_chunk_idx = idx - 1;
                break;
            }
            last_chunk = chunk.to_vec();
        }

        let mut chunked = Vec::new();
        for chunk in enc.chunks(16) {
            chunked.push(chunk);
        }

        // 2 chunks in front of the first A block will be the prepared 'admin + padding' block
        *chunked.last_mut().unwrap() = chunked[as_chunk_idx + 2];
        //println!("copied chunk at idx {}", as_chunk_idx + 2);

        let mut dechunked = Vec::new();
        for chunk in chunked {
            dechunked.append(&mut chunk.to_vec());
        }

        if check_success(&dechunked, &key) == true {
            println!("Challenge 13: Successful! Made a user with role=admin");
        } else {
            panic!("Challenge 13: failed!");
        }

    }

    /// So horribly bad code...
    pub fn challenge_14(){
        fn enc_oracle(arg_input: &Vec<u8>) -> Vec<u8>{
            let mut rng = helpers::Rand::new(12341337);
            // JEEEEEssus that is one long line
            //let mut rng = helpers::Rand::new(SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_micros() as u64);

            let prepend_count = rng.rand_u64() % 0x120;  // Cant have up to 2^64 random bytes common...
            let mut prepend_bytes = rng.rand_u8_vec(prepend_count as usize);

            let mut rng = helpers::Rand::new(1337); // constant seed to always get the same key
            let key = rng.rand_u8_vec(16);
            prepend_bytes.append(&mut arg_input.clone());

            // Append this unknown blob at the start
            let mut raw =
                helpers::b64_decode(&"Um9sbGluJyBpbiBteSA1LjAKV2l0aCBteSByYWctdG9wIGRvd24gc28gbXkgaGFpciBjYW4gYmxvdwpUaGUgZ2lybGllcyBvbiBzdGFuZGJ5IHdhdmluZyBqdXN0IHRvIHNheSBoaQpEaWQgeW91IHN0b3A/IE5vLCBJIGp1c3QgZHJvdmUgYnkK".to_string());

            // input = arg_input + secret
            //input.append(&mut raw);
            prepend_bytes.append(&mut raw);

            let output = helpers::aes_128_ecb_encrypt(&prepend_bytes, &key);

            return output;
            }
        fn index_of_control_start(enc: &[u8]) -> Option<usize> {
            let mut last_chunk = vec![0; 16];
            for (idx, chunk) in enc.chunks(16).enumerate() {
                if last_chunk == chunk {
                    return Some((idx - 1) * 16);
                }
                last_chunk = chunk.to_vec();
            }

            return None;
        }

        //println!("Doing challenge 14");
        print!("Challenge 14: ");
        let _ = stdout().flush();

        let mut res = Vec::new();
        let encrypted = enc_oracle(&b"A".to_vec());
        let encrypted_len = encrypted.len();
        //println!("encrypted({}): {:x?}", encrypted.len(), encrypted);

        // Find how many bytes we need to append to make the length increase.
        // that means we filled the entire block and openssl had to make another one with only
        // padding in it
        let mut req_size = 0;

        for i in 0..16 {
            let encrypted = enc_oracle(&"A".repeat(i).as_bytes().to_vec());
            if encrypted.len() != encrypted_len {
                req_size = i;
                break;
            }
        }

        let tmpenc = enc_oracle(&"A".repeat(16 * 4 - req_size).as_bytes().to_vec());
        let secret_len = index_of_control_start(&tmpenc).unwrap();
        let secret_len = tmpenc.len() - secret_len - 16 * 4 + 16 - req_size - 2;

        //println!("length of secret: {}", secret_len);
        // Make sure it's really ECB mode
        let needle = "A".repeat(16).repeat(4).as_bytes().to_vec();
        let mode = super::set1::is_ecb_encrypted(enc_oracle(&needle));

        if mode == false {
            panic!("Guessed wrong mode in challenge 14!");
        }

        //println!("req_size: {}", req_size);
        //println!("size of secret: {}", encrypted_len - 16);

        // Start of the actual logic
        // One byte short since we always want a byte from the secret part
        //let mut known_chunk = "A".repeat(((encrypted_len - req_size) / 16) * 16 + 16 - 1).to_string().as_bytes().to_vec();
        //let mut known_chunk = "A".repeat(128 + 0x10 - 1).to_string().as_bytes().to_vec();
        let mut known_chunk = "A".repeat(secret_len - 1 + 16 - req_size - 6 - 2).to_string().as_bytes().to_vec();
        let known_chunk_len = known_chunk.len();
        let mut precomp_block = "A".repeat(16 * 1).to_string().as_bytes().to_vec();
        let mut needle_plus_precomp = Vec::new();

        //assert!(known_chunk_len % 16 == 15);
        //println!("known_chunk_len: {}", known_chunk_len);

        // Only need to do this once
        let control_start = index_of_control_start(&enc_oracle(&known_chunk));
        let control_start = if let Some(v) = control_start {
            v
        } else {
            panic!("paniccccC!!");
        };

        //println!("enc - req: {}", encrypted_len - req_size);

        for _ in 0..secret_len {
            let mut lookup = HashMap::new();
            // Build up the mapping for all values

            for c in 1u8..127 {
                // Always want to change the last byte
                let mut arr = [0; 16];
                precomp_block[15] = c;
                //println!("\nprecomp_block: {:x?}", precomp_block);

                needle_plus_precomp.clear();
                needle_plus_precomp.extend("A".repeat(14).as_bytes().to_vec()); // No clue why 14 but it is
                needle_plus_precomp.extend(&needle);
                needle_plus_precomp.extend(&precomp_block);

                let encrypted = enc_oracle(&needle_plus_precomp);
                //println!("encrypted: ");
                //print_in_chunks(&encrypted);

                let control_start = index_of_control_start(&encrypted);
                let control_start = if let Some(v) = control_start {
                      v
                } else {
                    panic!("paniccccC!!");
                };
                let start_range = control_start + 16 * 4;
                let end_range = start_range + 16;
                //assert!(start_range % 16 == 0);
 
                //println!("control_start: {}\tusing range {}..{}", control_start, start_range, end_range);

                arr.copy_from_slice(&encrypted[start_range..end_range]);
                //println!("mapping {:x?} to -> {:x}", &encrypted[start_range..end_range], c);
                lookup.insert(arr, c);
            }

            let encrypted = enc_oracle(&known_chunk);


            //println!("control_start: {}", control_start);
            
            //println!("encrypted: ({})", encrypted.len());
            //print_in_chunks(&encrypted);

            let start_range = (control_start + known_chunk_len - 16) / 16 * 16;
            let end_range = (control_start + known_chunk_len) / 16 * 16;

            //println!("i: {}\tusing {}..{} as index", i, start_range, end_range);
            //println!("at index: {:x?}", &encrypted[start_range..end_range]);

            let mut tmparr = [0; 16];
            tmparr.copy_from_slice(&encrypted[start_range..end_range]);

            let last_byte = lookup.get(&tmparr);

            let last_byte = match last_byte {
                Some(val) => *val,
                None => {
                    println!("What we got so far: {}", std::str::from_utf8(&res).unwrap());
                    panic!("No such entry in the hashmap!")
                },
            };

            res.push(last_byte);

            if known_chunk.len() == 0 {
                println!("hit the if known_chunk.len break");
                break;
            }

            // Remove another byte so that we actually decrease in size and get
            // new bytes from the secret
            known_chunk.remove(0);

            precomp_block[15] = last_byte;
            precomp_block.rotate_left(1);
        }

        let secret_decoded = helpers::b64_decode(&"Um9sbGluJyBpbiBteSA1LjAKV2l0aCBteSByYWctdG9wIGRvd24gc28gbXkgaGFpciBjYW4gYmxvdwpUaGUgZ2lybGllcyBvbiBzdGFuZGJ5IHdhdmluZyBqdXN0IHRvIHNheSBoaQpEaWQgeW91IHN0b3A/IE5vLCBJIGp1c3QgZHJvdmUgYnkK".to_string());

        let resstr = std::str::from_utf8(&res).expect("Definitely something wrong!!");
        //println!("result: {}", resstr);
        assert!(resstr == std::str::from_utf8(&secret_decoded).unwrap());

        //println!("result: {}", resstr);
        println!("Successful! Decrypted with oracle!");

    }

    pub fn challenge_15(){
        match helpers::pkcs7_unpad(b"ICE ICE BABY\x04\x04\x04\x04".to_vec()){
            Ok(v) => assert!(v == b"ICE ICE BABY".to_vec()),
            Err(_) => panic!("pkcs unpad failed"),
        }

        match helpers::pkcs7_unpad(b"ICE ICE BABY\x05\x05\x05\x05".to_vec()){
            Ok(_) => panic!("invalid padding considered valid"),
            Err(_) => (),
        }

        match helpers::pkcs7_unpad(b"ICE ICE BABY\x01\x02\x03\x04".to_vec()){
            Ok(_) => panic!("invalid padding considered valid"),
            Err(_) => (),
        }

        match helpers::pkcs7_unpad(b"YELLOW SUBMARINE\x10\x10\x10\x10\x10\x10\x10\x10\x10\x10\x10\x10\x10\x10\x10\x10".to_vec()){
            Ok(v) => assert!(v == b"YELLOW SUBMARINE".to_vec()),
            Err(_) => panic!("pkcs unpad failed"),
        }

        println!("Challenge 15: Successful!");
    }

    // Went a bit overkill I realize now, didn't need to do a padding oracle attack.
    // Could have just flipped some bytes until it was correct lol.
    pub fn challenge_16(){
        fn encrypter(user_str: &str) -> Vec<u8> {
            let mut rng = helpers::Rand::new(1337);
            let key = rng.rand_u8_vec(16);
            let iv = rng.rand_u8_vec(16);

            let user_str = user_str.replace(";", "");
            let user_str = user_str.replace("=", "");

            let preappended = "comment1=cooking%20MCs;userdata=".to_string() + &user_str + ";comment2=%20like%20a%20pound%20of%20bacon";

            //println!("preappended: {}", preappended);

            helpers::aes_128_cbc_encrypt(preappended.as_bytes(), &key, &iv)
        }
        fn decrypter_oracle(ciphertext: &[u8]) -> Result<bool, String> {
            let mut rng = helpers::Rand::new(1337);
            let key = rng.rand_u8_vec(16);
            let iv = rng.rand_u8_vec(16);

            let decres = helpers::aes_128_cbc_decrypt(ciphertext, &key, &iv);
            let decrypted = match decres {
                Ok(v) => v,
                Err(e) => return Err(e),
            };

            //println!("decrypter_oracle: last byte of last block: {:#02x}", decrypted.last().unwrap());
            //println!("decrypter_oracle: decrypted: {:?}", decrypted);

            //let dec_str = std::str::from_utf8(&decrypted);
            // There will be completely scrambled blocks so need to force rust
            // to use it as a string anyway
            let dec_str = String::from_utf8_lossy(&decrypted);

            //println!("as string: {:x?}", dec_str);

            //let dec_str = match dec_str {
            //    Err(e) => return Ok(false),
            //    Ok(v) => v,
            //};

            //println!("decrypter_oracle: decrypted as string: {}", dec_str);

            if dec_str.contains(";admin=true;") {
                return Ok(true);
            } else {
                return Ok(false);
            }
        }
        print!("Challenge 16: ");
        let _ = stdout().flush();
        // Sanity checks
        let mut encd = encrypter("RealData123AadminAtrue");   // Eats invalid characters
        let decres = decrypter_oracle(&encd);
        assert!(decres.unwrap() == false);
        *encd.last_mut().unwrap() = 0xaa;
        let decres = decrypter_oracle(&encd);
        match decres {
            Err(_) => (),
            Ok(_) => panic!("baad"),
        }

        // Start of actual code
        // Goal is to swap the A's for ';' and '='
        let mut encd = encrypter("RealDataAadminAtrue");   // Eats invalid characters
        let orig_copy = encd.to_vec();
        
        let mut interm_state = vec![0u8; encd.len() - 16];
        let mut decrypted = vec![0u8; encd.len() - 16];

        // iterate over all blocks except for the last one
        for i in (0..encd.len() - 16).rev() {
            let orig_val = encd[i];
            let mut prev_orig_val = 0;

            if i > 1 {
                prev_orig_val = encd[i - 1];
                encd[i - 1] = encd[i - 1].overflowing_add(0x20).0;
            }

            let itend = i + (16 - i % 16);
            let itstart = i + 1;
            for j in itstart..itend {
                encd[j] = interm_state[j] ^ (16 - i % 16) as u8;
            }

            let bound = encd.len() - (encd.len() - 16 - i - 1) / 16 * 16;
            let mut found_byte = false;
            for b in 0..=255 {
                encd[i] = b;

                if let Ok(_) = decrypter_oracle(&encd[..bound]) {
                    interm_state[i] = b ^ (16 - i % 16) as u8;
                    decrypted[i] = orig_val ^ interm_state[i];
                    found_byte = true;
                }
            }

            encd[i] = orig_val;

            if !found_byte {
                panic!("Couldn't find valid padding for idx {}", i);
            }

            // Restore all the bytes we fiddled with
            if i > 1 {
                encd[i - 1] = prev_orig_val;
            }

            for j in itstart..itend {
                encd[j] = orig_copy[j];
            }

            assert!(orig_copy == encd);
        }

        let depadded = &helpers::pkcs7_unpad(decrypted).unwrap();
        //println!("depadded: {:x?}", depadded);
        let dec_str = std::str::from_utf8(&depadded).expect("Not done yet!");
        //println!("decrypted as string: {}", dec_str);

        let needle_pos = if let Some(v) = dec_str.find("AadminAtrue") {
            v
        } else {
            panic!("Got the wrong output from the padding oracle");
        };

        encd[needle_pos] = interm_state[needle_pos] ^ (';' as u8);  // 0x3d = '='
        encd[needle_pos + "admin".len() + 1] = interm_state[needle_pos + "admin".len() + 1] ^ ('=' as u8);  // 0x3d = '='

        if let Ok(true) = decrypter_oracle(&encd) {
            //println!("Whoop whoop! We did it!");
        } else {
            panic!("Not quite!");
        }

        println!("Successful! Decrypted and modified plaintext with CBC padding oracle!");
    }
}

mod set3 {
    use super::helpers;
    use std::time::{SystemTime, UNIX_EPOCH};
    use std::fs::File;
    use std::io::{BufRead, BufReader};
    //use std::{thread, time};
    use std::io::{stdout, Write};

    pub fn challenge_17() {
        /// Returns the ciphertext in .0 and iv in .1
        fn encrypter() -> (Vec<u8>, Vec<u8>){
            let mut rng = helpers::Rand::new(0x1337);
            let key = rng.rand_u8_vec(16);
            let iv = rng.rand_u8_vec(16);

            let mut rng = helpers::Rand::new(SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_micros() as u64);

            let secret_arr = ["MDAwMDAwTm93IHRoYXQgdGhlIHBhcnR5IGlzIGp1bXBpbmc=",
            "MDAwMDAxV2l0aCB0aGUgYmFzcyBraWNrZWQgaW4gYW5kIHRoZSBWZWdhJ3MgYXJlIHB1bXBpbic=",
            "MDAwMDAyUXVpY2sgdG8gdGhlIHBvaW50LCB0byB0aGUgcG9pbnQsIG5vIGZha2luZw==",
            "MDAwMDAzQ29va2luZyBNQydzIGxpa2UgYSBwb3VuZCBvZiBiYWNvbg==",
            "MDAwMDA0QnVybmluZyAnZW0sIGlmIHlvdSBhaW4ndCBxdWljayBhbmQgbmltYmxl",
            "MDAwMDA1SSBnbyBjcmF6eSB3aGVuIEkgaGVhciBhIGN5bWJhbA==",
            "MDAwMDA2QW5kIGEgaGlnaCBoYXQgd2l0aCBhIHNvdXBlZCB1cCB0ZW1wbw==",
            "MDAwMDA3SSdtIG9uIGEgcm9sbCwgaXQncyB0aW1lIHRvIGdvIHNvbG8=",
            "MDAwMDA4b2xsaW4nIGluIG15IGZpdmUgcG9pbnQgb2g=",
            "MDAwMDA5aXRoIG15IHJhZy10b3AgZG93biBzbyBteSBoYWlyIGNhbiBibG93"];

            let rand_secret = secret_arr[(rng.rand_u8() % secret_arr.len() as u8) as usize];
            let rand_secret = helpers::b64_decode(rand_secret);

            let encd = helpers::aes_128_cbc_encrypt(&rand_secret, &key, &iv);

            return (encd, iv);
        }
        fn decrypter_oracle(ciphertext: &[u8]) -> bool {
            let mut rng = helpers::Rand::new(0x1337);
            let key = rng.rand_u8_vec(16);
            let iv = rng.rand_u8_vec(16);

            let decres = helpers::aes_128_cbc_decrypt(ciphertext, &key, &iv);
            match decres {
                Ok(_) => return true,
                Err(_) => return false,
            };

        }
        // Start of actual code
        let tmp = encrypter();   // Eats invalid characters
        let iv = tmp.1;

        let mut encd = iv;
        encd.extend(tmp.0);

        let orig_copy = encd.to_vec();
        
        let mut interm_state = vec![0u8; encd.len() - 16];
        let mut decrypted = vec![0u8; encd.len() - 16];

        // iterate over all blocks except for the last one
        for i in (0..encd.len() - 16).rev() {
            let orig_val = encd[i];
            let mut prev_orig_val = 0;

            if i > 1 {
                prev_orig_val = encd[i - 1];
                encd[i - 1] = encd[i - 1].overflowing_add(0x20).0;
            }

            let itend = i + (16 - i % 16);
            let itstart = i + 1;
            for j in itstart..itend {
                encd[j] = interm_state[j] ^ (16 - i % 16) as u8;
            }

            let bound = encd.len() - (encd.len() - 16 - i - 1) / 16 * 16;
            let mut found_byte = false;
            for b in 0..=255 {
                encd[i] = b;

                if decrypter_oracle(&encd[..bound]) == true {
                    interm_state[i] = b ^ (16 - i % 16) as u8;
                    decrypted[i] = orig_val ^ interm_state[i];
                    found_byte = true;
                }
            }

            encd[i] = orig_val;

            if !found_byte {
                panic!("Couldn't find valid padding for idx {}", i);
            }

            // Restore all the bytes we fiddled with
            if i > 1 {
                encd[i - 1] = prev_orig_val;
            }

            for j in itstart..itend {
                encd[j] = orig_copy[j];
            }

            assert!(orig_copy == encd);
        }

        let dec_str = helpers::pkcs7_unpad(decrypted).unwrap();
        let dec_str = std::str::from_utf8(&dec_str).unwrap();

        //println!("decrypted as str: {}", dec_str);

        let res_strs = [
        "000000Now that the party is jumping",
        "000001With the bass kicked in and the Vega's are pumpin'",
        "000002Quick to the point, to the point, no faking",
        "000003Cooking MC's like a pound of bacon",
        "000004Burning 'em, if you ain't quick and nimble",
        "000005I go crazy when I hear a cymbal",
        "000006And a high hat with a souped up tempo",
        "000007I'm on a roll, it's time to go solo",
        "000008ollin' in my five point oh",
        "000009ith my rag-top down so my hair can blow",
        ];

        assert!(res_strs.contains(&dec_str));
        println!("Challenge 17: Successful!");
    }

    pub fn challenge_18() {
        let ciphertext = helpers::b64_decode("L77na/nrFsKvynd6HzOoG7GHTLXsTVu9qvY/2syLXzhPweyyMTJULu/6/kXX0KSvoOLSFQ==");

        let plaintext = helpers::aes_128_ctr_decrypt(&ciphertext, "YELLOW SUBMARINE".as_bytes(), 0);

        //println!("plaintext: {}", std::str::from_utf8(&plaintext).unwrap());
        assert!(plaintext == b"Yo, VIP Let's kick it Ice, Ice, baby Ice, Ice, baby ".to_vec());

        println!("Challenge 18: Successful!");
    }

    pub fn challenge_19() {
        let plain_arr = ["SSBoYXZlIG1ldCB0aGVtIGF0IGNsb3NlIG9mIGRheQ==",
"Q29taW5nIHdpdGggdml2aWQgZmFjZXM=", "RnJvbSBjb3VudGVyIG9yIGRlc2sgYW1vbmcgZ3JleQ==",
"RWlnaHRlZW50aC1jZW50dXJ5IGhvdXNlcy4=", "SSBoYXZlIHBhc3NlZCB3aXRoIGEgbm9kIG9mIHRoZSBoZWFk",
"T3IgcG9saXRlIG1lYW5pbmdsZXNzIHdvcmRzLA==", "T3IgaGF2ZSBsaW5nZXJlZCBhd2hpbGUgYW5kIHNhaWQ=",
"UG9saXRlIG1lYW5pbmdsZXNzIHdvcmRzLA==", "QW5kIHRob3VnaHQgYmVmb3JlIEkgaGFkIGRvbmU=",
"T2YgYSBtb2NraW5nIHRhbGUgb3IgYSBnaWJl", "VG8gcGxlYXNlIGEgY29tcGFuaW9u",
"QXJvdW5kIHRoZSBmaXJlIGF0IHRoZSBjbHViLA==", "QmVpbmcgY2VydGFpbiB0aGF0IHRoZXkgYW5kIEk=",
"QnV0IGxpdmVkIHdoZXJlIG1vdGxleSBpcyB3b3JuOg==", "QWxsIGNoYW5nZWQsIGNoYW5nZWQgdXR0ZXJseTo=",
"QSB0ZXJyaWJsZSBiZWF1dHkgaXMgYm9ybi4=", "VGhhdCB3b21hbidzIGRheXMgd2VyZSBzcGVudA==",
"SW4gaWdub3JhbnQgZ29vZCB3aWxsLA==", "SGVyIG5pZ2h0cyBpbiBhcmd1bWVudA==",
"VW50aWwgaGVyIHZvaWNlIGdyZXcgc2hyaWxsLg==", "V2hhdCB2b2ljZSBtb3JlIHN3ZWV0IHRoYW4gaGVycw==",
"V2hlbiB5b3VuZyBhbmQgYmVhdXRpZnVsLA==", "U2hlIHJvZGUgdG8gaGFycmllcnM/",
"VGhpcyBtYW4gaGFkIGtlcHQgYSBzY2hvb2w=", "QW5kIHJvZGUgb3VyIHdpbmdlZCBob3JzZS4=",
"VGhpcyBvdGhlciBoaXMgaGVscGVyIGFuZCBmcmllbmQ=", "V2FzIGNvbWluZyBpbnRvIGhpcyBmb3JjZTs=",
"SGUgbWlnaHQgaGF2ZSB3b24gZmFtZSBpbiB0aGUgZW5kLA==", "U28gc2Vuc2l0aXZlIGhpcyBuYXR1cmUgc2VlbWVkLA==",
"U28gZGFyaW5nIGFuZCBzd2VldCBoaXMgdGhvdWdodC4=", "VGhpcyBvdGhlciBtYW4gSSBoYWQgZHJlYW1lZA==",
"QSBkcnVua2VuLCB2YWluLWdsb3Jpb3VzIGxvdXQu", "SGUgaGFkIGRvbmUgbW9zdCBiaXR0ZXIgd3Jvbmc=",
"VG8gc29tZSB3aG8gYXJlIG5lYXIgbXkgaGVhcnQs", "WWV0IEkgbnVtYmVyIGhpbSBpbiB0aGUgc29uZzs=",
"SGUsIHRvbywgaGFzIHJlc2lnbmVkIGhpcyBwYXJ0", "SW4gdGhlIGNhc3VhbCBjb21lZHk7",
"SGUsIHRvbywgaGFzIGJlZW4gY2hhbmdlZCBpbiBoaXMgdHVybiw=", "VHJhbnNmb3JtZWQgdXR0ZXJseTo=",
"QSB0ZXJyaWJsZSBiZWF1dHkgaXMgYm9ybi4="];

        // Setup stuffs
        let mut decoded_arr: Vec<Vec<u8>> = Vec::new();
        for enc in plain_arr.iter() {
            decoded_arr.push(helpers::b64_decode(enc));
        }

        let mut rng = helpers::Rand::new(13371337);
        let key = rng.rand_u8_vec(16);
        let mut encrypted_arr: Vec<Vec<u8>> = Vec::new();
        for dec in decoded_arr {
            encrypted_arr.push(helpers::aes_128_ctr_encrypt(&dec, &key, 0));
        }

        // Ready for the actual logic now
        // Do detect single_key_xor for the first byte of all ciphertexts and
        // 2nd byte of all ciphertexts, and 3rd and so on
        let mut transposed: Vec<Vec<u8>> = Vec::new();

        let mut max_len = 0;

        for i in 0..encrypted_arr.len() {
            if encrypted_arr[i].len() > max_len {
                max_len = encrypted_arr[i].len();
            }
        }

        for _ in 0..max_len {
            transposed.push(Vec::new());
        }

        for i in 0..encrypted_arr.len() {
            for j in 0..encrypted_arr[i].len() {
                transposed[j].push(encrypted_arr[i][j]);
            }
        }

        let mut keystream = Vec::new();

        for b in transposed {
            let res = helpers::detect_single_key_xor(&b);
            keystream.push(res.2);
            //println!("res[{}]: {:x?}", idx, res);
        }

        //println!("keystream: {:x?}", keystream);

        let mut decrypted = Vec::new();

        for e in &encrypted_arr {
            let res = helpers::repeating_key_xor(&keystream, e);
            let res = std::str::from_utf8(&res).unwrap().to_string();
            //println!("res {} ({}): '{}'", idx, res.len(), res);
            decrypted.push(res);
        }

        // Not perfect output but good enough
        assert!(decrypted[1] == "Coming with vivid faces");
        println!("Challenge 19: Successful! Broke CTR mode with fixed nonce");

    }

    pub fn challenge_20() {
        let file = File::open("20.txt").expect("Failed opening 20.txt");
        let reader = BufReader::new(file);
        let mut rng = helpers::Rand::new(13371337);
        let key = rng.rand_u8_vec(16);

        let mut encrypted_arr: Vec<Vec<u8>> = Vec::new();

        // Take the plaintexts and decode and encrypt them
    
        // Read file line by line
        for l in reader.lines() {
            let l = l.unwrap();
            encrypted_arr.push(helpers::aes_128_ctr_encrypt(&helpers::b64_decode(&l), &key, 0));
        }

        //println!("encrypted_arr: {:x?}", encrypted_arr);

        // Ready for the actual logic now
        // Do detect single_key_xor for the first byte of all ciphertexts and
        // 2nd byte of all ciphertexts, and 3rd and so on
        let mut transposed: Vec<Vec<u8>> = Vec::new();

        let mut max_len = 0;

        for i in 0..encrypted_arr.len() {
            if encrypted_arr[i].len() > max_len {
                max_len = encrypted_arr[i].len();
            }
        }

        for _ in 0..max_len {
            transposed.push(Vec::new());
        }

        //println!("transposed len: {}", transposed.len());

        for i in 0..encrypted_arr.len() {
            for j in 0..encrypted_arr[i].len() {
                transposed[j].push(encrypted_arr[i][j]);
            }
        }

        let mut keystream = Vec::new();

        for b in transposed {
            let res = helpers::detect_single_key_xor(&b);
            keystream.push(res.2);
            //println!("res: {:x?}", res);
        }

        //println!("keystream: {:x?}", keystream);

        let mut decrypted = Vec::new();

        for e in &encrypted_arr {
            let res = helpers::repeating_key_xor(&keystream, e);
            let res = std::str::from_utf8(&res).unwrap().to_string();
            //println!("res ({}): '{}'", res.len(), res);
            decrypted.push(res);
        }

        // First byte is slightly off on all of them...
        // but reaaally close though

        assert!(decrypted[55] == " Cause my girl is definitely mad / 'Cause it took us too long to do this album");
        println!("Challenge 20: Successful! Broke CTR mode with fixed nonce");

    }

    pub fn challenge_21(){
        let res_arr: [u32; 10] = [
            1791095845,
            4282876139,
            3093770124,
            4005303368,
            491263,
            550290313,
            1298508491,
            4290846341,
            630311759,
            1013994432,
        ];
        let mut rng = helpers::RandMT::new(1);

        for i in 0..10 {
            assert!(rng.rand_u32() == res_arr[i])
        }

        rng.seed(1);

        for i in 0..10 {
            assert!(rng.rand_u32() == res_arr[i])
        }

        println!("Challenge 21: Successful! Working implementation of Mersenne twister");
    }

    // Time stuff in rust results in horribly long statements.
    pub fn challenge_22() {
        let mut rng = helpers::Rand::new(SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_micros() as u64);
        print!("Challenge 22: ");
        let _ = stdout().flush();

        let now = SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_secs() as u32;

        let rand_sleep = (rng.rand_u32() % 1000) + 1;
        //println!("sleeping for {} seconds", rand_sleep);

        //thread::sleep(time::Duration::from_secs(rand_sleep as u64));
        let now = now + rand_sleep;

        let rand_sleep = (rng.rand_u32() % 1000) + 1;
        //let seed_val = SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_secs() as u32;
        let seed_val = now;
        let mut rngmt = helpers::RandMT::new(seed_val);
        //println!("sleeping for {} more seconds", rand_sleep);
        //thread::sleep(time::Duration::from_secs(rand_sleep as u64));
        let now = now + rand_sleep;

        let randval = rngmt.rand_u32();

        // Okey. Now figure out the seed from randval...
        //let now = SystemTime::now();
        let now = now + rand_sleep; // Emulate time passing
        //let now_secs = now.duration_since(UNIX_EPOCH).unwrap().as_secs() as u32;
        let now_secs = now;
        
        // Not really, but in the spirit of things
        //let then_start = now.checked_sub(Duration::from_secs(1500 * 2)).unwrap().duration_since(UNIX_EPOCH).unwrap().as_secs() as u32;  
        let then_start = (now - (1500 * 2)) as u32;  

        let mut tmprng = helpers::RandMT::new(1);
        let mut seed_guess = 0;

        for secs in then_start..=now_secs {
            tmprng.seed(secs);

            if tmprng.rand_u32() == randval {
                //println!("Got the seed!");
                //println!("Seed guess: {}\nActual seed: {}", secs, seed_val);
                seed_guess = secs;
                break;
            }
        }

        assert!(seed_guess == seed_val);

        println!("Successful! Cracked time based seed");

    }

    pub fn challenge_23() {
        fn temper(y: u32) -> u32 {
            let y1 = y ^ (y >> helpers::RandMT::U);
            let y2 = y1 ^ ((y1 << helpers::RandMT::S) & helpers::RandMT::B);
            let y3 = y2 ^((y2 << helpers::RandMT::T) & helpers::RandMT::C);
            let y4 = y3 ^ (y3 >> helpers::RandMT::L);
            y4
        }
        
        let y = 0x5A5A5A5A;
        let tempered = temper(y);
        assert!(y == helpers::mt_untemper(tempered));
        let y = 0xA5A5A5A5;
        let tempered = temper(y);
        assert!(y == helpers::mt_untemper(tempered));
        let y = 0x41414141;
        let tempered = temper(y);
        assert!(y == helpers::mt_untemper(tempered));
        let y = 0x12345678;
        let tempered = temper(y);
        assert!(y == helpers::mt_untemper(tempered));

        let mut rng = helpers::RandMT::new(1234);
        let mut first_624 = Vec::new();
        let mut untempered = Vec::new();

        for _ in 0..624 {
            first_624.push(rng.rand_u32());
        }

        for i in 0..624 {
            untempered.push(helpers::mt_untemper(first_624[i]));
        }

        let mut rng_clone = helpers::RandMT::from_state(&untempered);

        for _ in 0..624 {
            assert!(rng_clone.rand_u32() == rng.rand_u32());
        }

        println!("Challenge 23: Successful! Cloned Mersenne twister RNG state from 624 outputs");
    }

    /// Generate a 'password reset code' seeded by the current time
    fn gen_pass_reset_code_impl(seed: u32) -> String {
        let tokens = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789";
        //let seed_val = SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_secs() as u32;
        let mut rng = helpers::RandMT::new(seed);
        let mut reset_code = Vec::new();

        // 14 is a safe length for a password reset code
        for _ in 0..14 {
            let index = (rng.rand_u8() % tokens.len() as u8) as usize;
            let code = tokens[index];
            reset_code.push(code);
        }

        let reset_code = std::str::from_utf8(&reset_code).unwrap().to_string();
        //println!("Reset code: {:x?}", &reset_code);

        return reset_code;
    }

    fn gen_pass_reset_code() -> String {
        let seed_val = SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_secs() as u32;

        return gen_pass_reset_code_impl(seed_val);
    }

    fn is_valid_reset_code(code: &str) -> bool {
        let start_time = (SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_secs() as u32) - 60;
        let now = SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_secs() as u32;

        for time in start_time..=now {
            if code == gen_pass_reset_code_impl(time) {
                //println!("Was a valid reset code");
                return true;
            }
        }
        return false;
    }

    pub fn challenge_24() {
        let mut rng = helpers::RandMT::new(1337);
        let keystream1 = rng.rand_u8_vec(16);
        rng.seed(1337);
        let keystream2 = rng.rand_u8_vec(16);
        assert!(keystream1 == keystream2);

        // Check that the stream cipher works correctly
        let plaintext = b"YELLOW SUBMARINE 1234";
        let crypted = helpers::RandMT::stream_cipher_crypt(plaintext, 1337);
        assert!(&helpers::RandMT::stream_cipher_crypt(&crypted, 1337) == plaintext);

        // Encrypt a known plaintext prefixed with random count of random bytes.
        // Find the seed.
        let mut rng = helpers::RandMT::new(1337);
        let rand_count = rng.rand_u32() % 10;
        let mut rand_byte = rng.rand_u8_vec(rand_count as usize);

        rand_byte.extend("A".repeat(14).as_bytes());
        rng.seed(1337);
        let seed = rng.rand_u32() as u16;
        let mut seed_guess = 0;
        let crypted = helpers::RandMT::stream_cipher_crypt(&rand_byte, seed);
        let mut found_seed = false;

        for guess in 0..std::u16::MAX {
            let cguess = helpers::RandMT::stream_cipher_crypt(&crypted, guess);

            let len = cguess.len();
            if &cguess[len - 14..] == "A".repeat(14).as_bytes() {
                //println!("Found seed: {}", guess);
                found_seed = true;
                seed_guess = guess;
                break;
            }
        }
    
        assert!(seed == seed_guess);

        if !found_seed {
            panic!("Challenge 24: Failed! Couldn't find seed!");
        }

        // Carry on with the rest of the challenge
        let reset_code = gen_pass_reset_code();
        assert!(is_valid_reset_code(&reset_code) == true);

        let bs = reset_code.as_bytes();
        // Flip the first byte and it should be invalid
        let reset_code = (bs[0] ^ 0xa5).to_string() + &reset_code[1..];

        assert!(is_valid_reset_code(&reset_code) == false);
        println!("Challenge 24: Successful! Bad Mersenne twister stream cipher broken");

    }
}

mod set4 {
    use super::helpers;
    use std::io::{stdout, Write};
    use std::time::{SystemTime};
    use std::{thread, time};
    use std::time::Duration;

    //use super::kv_parse as kv_parse;

    /// Doesn't expect offset to be 0 indexed.
    /// Doesn't return anything, but mutates the ciphertext in first argument.
    fn edit(ciphertext: &mut Vec<u8>, key: &[u8], nonce: u64, offset: usize, newtext: &[u8]) {
        let newlen = newtext.len();
        let cipherlen = ciphertext.len();
        let ctr_start: u64 = (offset / 16) as u64;
        let byte_offset = offset % 16;

        //println!("offset: {}\tnewlen: {}\tcipherlen: {}", offset, newlen, cipherlen);
        if offset + newlen > cipherlen {
            panic!("Tried to edit outside of the existing ciphertext length");
        }
        //println!("newtext: {:x?}", newtext);


        //println!("decrypting bytes at range {}..{}", (ctr_start * 16) as usize, (ctr_start * 16) as usize + newlen);
        let mut decrypted = helpers::aes_128_ctr_crypt_with_start_counter(
            &ciphertext[(ctr_start * 16) as usize..(ctr_start * 16) as usize + newlen + byte_offset], key, nonce, ctr_start);

        //println!("decrypted len: {}\tnewlen: {}", decrypted.len(), newlen);
        assert!(decrypted.len() == newlen + byte_offset);

        //println!("decrypted bytes at offset {}: {:x?}", offset, &decrypted[byte_offset..]);
        //let as_str = std::str::from_utf8(&decrypted).expect("Something went bad in edit");
        //println!("decrypted bytes at offset {}: \"{}\"", offset, &as_str[byte_offset..byte_offset + newlen]);

        let chunk_of_interest = &mut decrypted[byte_offset..];
        //println!("chunk_of_interest len: {}\nbyte_offset: {}", chunk_of_interest.len(), byte_offset);
        //println!("chunk_of_interest: {:x?}", chunk_of_interest);

        for (idx, c) in newtext.iter().enumerate() {
            //println!("idx: {}\tc: {:x}", idx, c);
            chunk_of_interest[idx] = *c;
        }

        //println!("chunk_of_interest after: {:x?}", chunk_of_interest);
        //println!("decrypted after: {:x?}", &decrypted[byte_offset..]);

        let re_encrypted = helpers::aes_128_ctr_crypt_with_start_counter(&decrypted, key, nonce, ctr_start);
        //println!("re_encrypted({}): {:x?}", re_encrypted.len(), re_encrypted);

        for (i, j) in (offset..offset + newlen).zip(byte_offset..byte_offset + newlen) {
            //println!("i: {}\tj: {}", i, j);
            ciphertext[i as usize] = re_encrypted[j as usize];
        }
    }

    pub fn challenge_25() {
        let raw = helpers::read_in_entire_file("25.txt");
        let as_str = std::str::from_utf8(&raw).unwrap();
        let raw = as_str.replace("\n", "");
        let raw = helpers::b64_decode(&raw);
        let raw = helpers::aes_128_ecb_decrypt(&raw, b"YELLOW SUBMARINE");

        let mut rng = helpers::Rand::new(1337);
        let key = rng.rand_u8_vec(16);

        let mut ciphertext = helpers::aes_128_ctr_encrypt(&raw, &key, 0);
        let orig_ciphertext = ciphertext.clone();

        //edit(&mut ciphertext, &key, 0, 1, b"123456789abcde");

        // The easy part left now.
        let known_as = "A".repeat(ciphertext.len()).into_bytes();
        let mut keystream = Vec::with_capacity(known_as.len());

        edit(&mut ciphertext, &key, 0, 0, &known_as);

        for i in 0..known_as.len() {
            keystream.push(ciphertext[i] ^ known_as[i]);
        }

        let mut decrypted = Vec::with_capacity(known_as.len());

        for i in 0..known_as.len() {
            decrypted.push(orig_ciphertext[i] ^ keystream[i]);
        }

        assert!(decrypted == raw);
        println!("Challenge 25: Successful! Broke AES CTR mode with edit function");

    }

    // The correct way this time
    pub fn challenge_26(){
        fn encrypter(user_str: &str) -> Vec<u8> {
            let mut rng = helpers::Rand::new(1337);
            let key = rng.rand_u8_vec(16);

            let user_str = user_str.replace(";", "");
            let user_str = user_str.replace("=", "");

            let preappended = "comment1=cooking%20MCs;userdata=".to_string() + &user_str + ";comment2=%20like%20a%20pound%20of%20bacon";

            //println!("preappended: {}", preappended);

            helpers::aes_128_ctr_encrypt(preappended.as_bytes(), &key, 0)
        }
        fn decrypter_oracle(ciphertext: &[u8]) -> Result<bool, String> {
            let mut rng = helpers::Rand::new(1337);
            let key = rng.rand_u8_vec(16);

            let decrypted = helpers::aes_128_ctr_decrypt(ciphertext, &key, 0);
            //let decrypted = match decres {
            //    Ok(v) => v,
            //    Err(e) => return Err(e),
            //};

            //println!("decrypter_oracle: last byte of last block: {:#02x}", decrypted.last().unwrap());
            //println!("decrypter_oracle: decrypted: {:?}", decrypted);

            // There will be completely scrambled blocks so need to force rust
            // to use it as a string anyway
            let dec_str = String::from_utf8_lossy(&decrypted);

            //println!("decrypter_oracle: decrypted as string: {}", dec_str);

            if dec_str.contains(";admin=true;") {
                return Ok(true);
            } else {
                return Ok(false);
            }
        }
        print!("Challenge 26: ");
        let _ = stdout().flush();

        // Start of actual code
        // Goal is to swap the A's for ';' and '='
        let mut encd = encrypter("RealDataAadminAtrue");   // Eats invalid characters
        
        // TODO: do the magic here
        let tmp = b"comment1=cooking%20MCs;userdata=RealDataABC";
        //println!("tmp: {:x}", &tmp["comment1=cooking%20MCs;userdata=".len() + "RealData".len()]);
        assert!(tmp["comment1=cooking%20MCs;userdata=".len() + "RealData".len()] == 0x41);
        let mut got_it = false;

        for guess in 0..std::u16::MAX {
            let [b1, b2] = guess.to_le_bytes();

            encd["comment1=cooking%20MCs;userdata=".len() + "RealData".len()] = b1;
            encd["comment1=cooking%20MCs;userdata=".len() + "RealDataAadmin".len()] = b2;

            if let Ok(true) = decrypter_oracle(&encd) {
                got_it = true;
                break;
            }
        }

        if !got_it {
            panic!("Failed!");
        }

        println!("Successful! Created an admin profile by bitflipping AES CTR mode ciphertext");
    }
    
    pub fn challenge_27() {
        fn encrypter(user_str: &str) -> Vec<u8> {
            let key = "YELLOW SUBMARINE".as_bytes();
            let iv = &key;

            let user_str = user_str.replace(";", "");
            let user_str = user_str.replace("=", "");

            //println!("preappended: {}", preappended);

            helpers::aes_128_cbc_encrypt(user_str.as_bytes(), &key, iv)
        }
        fn decrypter_oracle(ciphertext: &[u8]) -> Result<String, Vec<u8>> {
            let key = "YELLOW SUBMARINE".as_bytes();
            let iv = &key;

            let decres = helpers::aes_128_cbc_decrypt_no_unpad(ciphertext, &key, iv);
        
            let unpadded = helpers::pkcs7_unpad(decres.clone());

            let decrypted = match unpadded {
                Ok(v) => v,
                Err(_) => return Err(decres),
            };

            let dec_str = std::str::from_utf8(&decrypted);

            match dec_str {
                Ok(v) => return Ok(v.to_string()),
                Err(_) => return Err(decrypted),
            }
        }

        let enc_str = "A".repeat(16 * 3);
        let mut encd = encrypter(&enc_str);

        /*************************
        * Start of attacker code *
        **************************/

        let zeroes = vec![0; 16];
        encd.splice(16..32, zeroes.iter().cloned());
        let mut first_16 = vec![0; 16];
        first_16.clone_from_slice(&encd[..16]);

        encd.splice(32..32 + 16, first_16.iter().cloned());
        let attacker_modified = &encd;

        /*************************
        * End of attacker code *
        **************************/

        // Start of receiver code
        let decres = decrypter_oracle(attacker_modified);

        let dec_err = match decres {
            Ok(_) => panic!("Not supposed to be ok"),
            Err(v) => v,
        };

        /*******************************
        * Start of attacker code again *
        *******************************/

        let p1 = &dec_err[..16];
        let p2 = &dec_err[32..32+16];

        let key = helpers::repeating_key_xor(p1, p2);
        //let as_str = std::str::from_utf8(&key).unwrap();
        //println!("xored as string: \"{}\"", as_str);

        assert!(key == b"YELLOW SUBMARINE");
        println!("Challenge 27: Successful! Got key from MitM AES CBC key=IV scenario");
    }

    pub fn challenge_28() {
        let secret = b"SUPA SEKRIT";
        let message = b"YELLOW SUBMARINE";
        let mac = helpers::create_sha1_mac(secret, message);
        //println!("mac: {:x?}", mac);
        //println!("verified: {}", helpers::verify_mac(&mac, secret, message));
        assert!(helpers::verify_sha1_mac(&mac, secret, message));

        let message2 = b"ORANGE SUBMARINE";
        //println!("verified: {}", helpers::verify_mac(&mac, secret, message2));
        assert!(!helpers::verify_sha1_mac(&mac, secret, message2));

        let secret2 = b"BAD GUESS";
        //println!("verified: {}", helpers::verify_mac(&mac, secret2, message));
        assert!(!helpers::verify_sha1_mac(&mac, secret2, message));

        println!("Challenge 28: Successful! Have a basic SHA-1 MAC working");

    }

    pub fn challenge_29() {
        let secret = "SUPASUPA SEKR1T".as_bytes();
        let msg = "comment1=cooking%20MCs;userdata=foo;comment2=%20like%20a%20pound%20of%20bacon".as_bytes();
        let mac = helpers::create_sha1_mac(&secret, &msg);

        let mut new_msg = Vec::new();
        new_msg.extend(msg);
        new_msg.extend(helpers::sha1_padding_for(&msg));
        new_msg.extend(";admin=true".as_bytes());

        let tmpsum = helpers::sha1sum(&msg);
        let tmpstate = helpers::sha1_state_from(tmpsum);
        let cont_sum = helpers::sha1_from_state(tmpstate, b";admin=true", (((msg.len() / 64) + (msg.len() % 64 != 0) as usize) * 64) as u64);
        
        let mut tmp = vec![0; msg.len()];
        tmp.copy_from_slice(&msg);
        tmp.extend(helpers::sha1_padding_for(&msg));
        tmp.extend(";admin=true".as_bytes());

        let real_sum = helpers::sha1sum(&tmp);
        assert!(real_sum == cont_sum);
        //println!("Have a working length extension attack!");

        assert!(helpers::verify_sha1_mac(&mac, &secret, &msg) == true);
        let start_state = helpers::sha1_state_from(mac);
        //println!("mac: {:x?}", mac);

        let mut suc = false;

        //let mut tmp = vec![0; msg.len()];
        let mut tmp = Vec::new();
        tmp.extend(secret);
        tmp.extend(msg);

        let mut tmp2 = Vec::new();
        tmp2.extend(secret);
        tmp2.extend(msg);

        tmp.extend(helpers::sha1_padding_for(&tmp2));
        tmp.extend(";admin=true".as_bytes());

        for i in 0..40 {
            let new_mac = helpers::sha1_from_state(
                    start_state,
                    ";admin=true".as_bytes(),
                    ((((msg.len() + i) / 64) + ((msg.len() + i) % 64 != 0) as usize) * 64) as u64
                );


            let mut l_new_msg = Vec::new();
            let tmp_secret = vec![0x41; i as usize];    // Modeling secret
            l_new_msg.extend(msg);

            let mut tmp_pad = Vec::new();
            tmp_pad.extend(&tmp_secret);
            tmp_pad.extend(msg);

            l_new_msg.extend(helpers::sha1_padding_for(&tmp_pad));
            l_new_msg.extend(";admin=true".as_bytes());

            if helpers::verify_sha1_mac(&new_mac, &secret, &l_new_msg) == true {
                let tmpstr = String::from_utf8_lossy(&l_new_msg);
                if tmpstr.contains(";admin=true") {
                    suc = true;
                } else {
                    panic!("FAAIL! but with correct mac ...?");
                }
            }
        }


        assert!(suc);
        println!("Challenge 29: Successful! Forged a SHA1 mac with a length extension");
    }

    /// MD4 has some weird endianness stuff, jeeeez
    pub fn challenge_30() {
        {
        // Some sanity checks
        let secret = b"SUPA SEKRIT";
        let message = b"YELLOW SUBMARINE";
        let mac = helpers::create_md4_mac(secret, message);
        assert!(helpers::verify_md4_mac(&mac, secret, message));
        let message2 = b"ORANGE SUBMARINE";
        assert!(!helpers::verify_md4_mac(&mac, secret, message2));
        let secret2 = b"BAD GUESS";
        assert!(!helpers::verify_md4_mac(&mac, secret2, message));
        }
        
        let secret = "SUPASUPA SEKR1T".as_bytes();
        let msg = "comment1=cooking%20MCs;userdata=foo;comment2=%20like%20a%20pound%20of%20bacon".as_bytes();
        let mac = helpers::create_md4_mac(&secret, &msg);

        let start_state = helpers::md4_state_from(mac);
        //println!("mac: {:x?}", mac);

        let mut suc = false;

        //let mut tmp = vec![0; msg.len()];
        let mut tmp = Vec::new();
        tmp.extend(secret);
        tmp.extend(msg);

        let mut tmp2 = Vec::new();
        tmp2.extend(secret);
        tmp2.extend(msg);

        tmp.extend(helpers::md4_padding_for(&tmp2));
        tmp.extend(";admin=true".as_bytes());

        for i in 0..40 {
            let new_mac = helpers::md4_from_state(
                    start_state,
                    ";admin=true".as_bytes(),
                    (((msg.len() + i) / 64) + ((msg.len() + i) % 64 != 0) as usize) * 64
                );


            let mut l_new_msg = Vec::new();
            let tmp_secret = vec![0x41; i as usize];    // Modeling secret
            l_new_msg.extend(msg);

            let mut tmp_pad = Vec::new();
            tmp_pad.extend(&tmp_secret);
            tmp_pad.extend(msg);

            l_new_msg.extend(helpers::md4_padding_for(&tmp_pad));
            l_new_msg.extend(";admin=true".as_bytes());

            if helpers::verify_md4_mac(&new_mac, &secret, &l_new_msg) == true {
                let tmpstr = String::from_utf8_lossy(&l_new_msg);
                if tmpstr.contains(";admin=true") {
                    suc = true;
                } else {
                    panic!("FAAIL! but with correct mac ...?");
                }
            }
        }

        assert!(suc);
        println!("Challenge 30: Successful! Forged an MD4 mac with a length extension");
    }

    /// Takes a string as file argument and checks if the signature is valid for it
    fn web_request(file: &str, signature: &[u8; 20]) -> bool {
        fn insecure_compare(sig1: &[u8; 20], sig2: &[u8; 20]) -> bool {
            for i in 0..20 {
                if sig1[i] != sig2[i] {
                    //println!("Expected: {:x?}\tgot: {:x?}", sig1, sig2);
                    return false;
                }
                thread::sleep(time::Duration::from_millis(10));
            }
            return true;
        }

        let hmac_key = "supa secret key".as_bytes();
        let hmac = helpers::create_sha1_hmac(hmac_key, file.as_bytes());

        return insecure_compare(&hmac, signature);
    }

    /// Takes like 30 minutes to run... insane...
    /// Challenge 31 and 32
    pub fn challenge_31() {
        const TRIES_PER_BYTE: usize = 5;
        let secret = "key".as_bytes();
        let msg = "The quick brown fox jumps over the lazy dog".as_bytes();
        let hmac = helpers::create_sha1_hmac(secret, msg);
        //println!("hmac for H(\"{:x?}\", \"{:x?}\") = {:x?}", secret, msg, hmac);
        assert!(helpers::verify_sha1_hmac(&hmac, secret, msg) == true);

        // Check the web_requst works correctly
        {
        let hmac_key = "supa secret key".as_bytes();
        let mesg = "filename";
        let hmac = helpers::create_sha1_hmac(hmac_key, mesg.as_bytes());
        let response = web_request(mesg, &hmac);
        assert!(response == true);
        }

        println!("Challenge 31: ");
        let _ = stdout().flush();

        // Start of actual challenge
        let wanted_file = "/etc/passwd";
        let hmac_key = "supa secret key".as_bytes();
        let wanted_hash = helpers::create_sha1_hmac(hmac_key, wanted_file.as_bytes());
        //print!("want: ");
        //for b in &wanted_hash {
        //    print!("{:02x} ", b);
        //}
        //println!("");

        let mut working_hmac = [0; 20];

        //print!("have: ");

        for i in 0..20 {
            let mut done = false;
            //println!("Working on byte {}", i);
            //let mut timings = 0u128;
            let mut timings = Duration::new(0, 0);
            let mut most_likely_byte = 0;
            for j in 0..=255 {
                working_hmac[i] = j;

                //let mut duration = 0;
                let mut duration = Duration::new(0, 0);

                for _ in 0..TRIES_PER_BYTE {

                    // Could try rdtsc aswell...
                    let start = SystemTime::now();
                    let r = web_request(wanted_file, &working_hmac);
                    let end = SystemTime::now();

                    if r {
                        working_hmac[i] = j;
                        done = true;
                        break;
                    }

                    let tmp = end.duration_since(start).unwrap();
                    // Or maybe not as micros or nanos, and use duration to do the calculation
                    duration += tmp;
                    //duration += tmp.as_nanos();    // or micros?
                    //duration += tmp.as_micros();
                }

                duration = duration / TRIES_PER_BYTE as u32;

                if done == true {
                    break;
                }

                if duration > timings {
                    timings = duration;
                    most_likely_byte = j;
                }
            }

            if done == true {
                break;
            }

            working_hmac[i] = most_likely_byte;
            print!("{:02x} ", most_likely_byte);
            let _ = stdout().flush();

            if working_hmac[i] != wanted_hash[i] {
                panic!("Failed!");
            }
        }

        //println!("");


        let res = web_request(wanted_file, &working_hmac);
        //println!("res: {}", res);
        assert!(res == true);

        println!("Successful!");
        //println!("Challenge 31: Successful!");
        println!("Challenge 32: Successful!");
    }
}

mod generic_kv_parse {
    use std::collections::HashMap;
    pub fn deserialize(serialized: &str) -> Option<HashMap<&str, String>> {
        let mut serialized = serialized.clone();
        let mut kvs = Vec::new();
        //let mut prof: Profile = Profile{email: "".to_string(), uid: 0, role: "".to_string()};
        let mut obj = HashMap::new();

        loop {
            let start = serialized.find("&");

            let start = match start {
                Some(val) => val,
                None => {
                    kvs.push(&serialized[..]);
                    break;
                },
            };

            kvs.push(&serialized[..start]);
            serialized = &serialized[start+1..];
        }
        //println!("key value pairs: {:?}", kvs);

        for item in kvs {
            //println!("item: {}", item);
            let name_offset = item.find("=");

            let name_offset = if let Some(val) = name_offset {
                val
            } else {
                panic!("AAAAHHH!!");
            };

            let name = &item[..name_offset];
            //println!("name: {}", &name);

            let value = &item[name_offset + 1..];


            obj.insert(name, value.to_string());

            //println!("value: {}\n", &item[name_offset + 1..]);
        }

        return Some(obj);
    }

    /*
    pub fn serialize(prof: &Profile) -> String {
        // No iterators for structs :(. Actually this is way more nice than an iterator.
        format!("email={}&uid={}&role={}", prof.email, prof.uid, prof.role)
    }
    */

    /*
    /// Returns the serialized form of the Profile for the email
    pub fn profile_for(email: &str) -> String {
        let prof = Profile {
            email: email.to_string(),
            uid: 10,
            role: "user".to_string(),
        };

        return serialize(&prof);
    }

    pub fn sanity_checks() {
        // Some sanity checks
        match deserialize("email=foo@bar.com&uid=10&role=user") {
            //Some(_) => println!("correctly serialized object succeeded"),
            Some(_) => (),
            None => panic!("correctly formatted serialized object failed!"),
        }
        match deserialize("email=f=o@bar.com&uid=10&role=user") {
            //Some(_) => panic!("incorrectly formatted serialized object succeeded!"),
            Some(_) => panic!("incorrectly formatted serialized object succeeded!"),
            None => (),
        }
        let serialized = "email=foo@bar.com&uid=10&role=user";
        let prof = deserialize(serialized);
        let prof = match prof {
            Some(val) => val,
            None => panic!("failed deserializing"),
        };
        let round_trip = serialize(&prof);
        //println!("round_trip: {}", round_trip);
        assert!(serialized == &round_trip);
    }
    */
}

mod set5 {
    use num_bigint::{BigUint, RandomBits};
    use rand::{Rng, RngCore};
    use super::helpers;
    use std::convert::TryInto;
    use super::generic_kv_parse;

    fn basic_dh() {
        let p = 37u64;
        let g = 5u64;

        //let a = rand::random::<u32>() % p as u32;
        let a = 7;
        let A = g.checked_pow(a).unwrap() % p;
        //println!("random number a: {}", a);
        //println!("A: {}", A);

        //let b = rand::random::<u32>() % p as u32;
        let b = 11;
        let B = g.checked_pow(b).unwrap() % p;
        //println!("random number b: {}", b);
        //println!("B: {}", B);

        let s1 = B.checked_pow(a).unwrap() % p;
        let s2 = A.checked_pow(b).unwrap() % p;

        assert!(s1 == s2, "basic DH logic failed! s1({}) != s2({})", s1, s2);

        //println!("shared secret: {}", s1);

        let s_key = helpers::sha1sum(&s1.to_le_bytes());

        //println!("shared session key: {:x?}", s_key);
    }

    fn gen_privkey() -> BigUint {
        let mut rng = rand::thread_rng();

        rng.sample(RandomBits::new(1024))
    }

    fn get_pubkey(p: &BigUint, g: &BigUint, a: &BigUint) -> BigUint {
        g.modpow(a, p) // A = (g**a) % p
    }

    fn gen_shared_session_key(A: &BigUint, b: &BigUint, p: &BigUint) -> [u8; 16] {
        let s = A.modpow(b, p);

        helpers::sha1sum(&s.to_bytes_le())[0..16].try_into().unwrap()
    }

    const P_STR: &[u8; 384] = b"ffffffffffffffffc90fdaa22168c234c4c6628b80dc1cd129024\
                    e088a67cc74020bbea63b139b22514a08798e3404ddef9519b3cd\
                    3a431b302b0a6df25f14374fe1356d6d51c245e485b576625e7ec\
                    6f44c42e9a637ed6b0bff5cb6f406b7edee386bfb5a899fa5ae9f\
                    24117c4b1fe649286651ece45b3dc2007cb8a163bf0598da48361\
                    c55d39a69163fa8fd24cf5f83655d23dca3ad961c62f356208552\
                    bb9ed529077096966d670c354e4abc9804f1746c08ca237327fff\
                    fffffffffffff";

    pub fn challenge_33() {
        basic_dh();

        let p = BigUint::parse_bytes(P_STR, 16).unwrap();
        let g = BigUint::from(2u32);

        //println!("p: {:x}", p);

        let a = gen_privkey();
        let A = get_pubkey(&p, &g, &a);

        let b = gen_privkey();
        let B = get_pubkey(&p, &g, &b);

        //println!("pubkey A: {:x}", A);
        //println!("pubkey B: {:x}", B);

        let s1 = gen_shared_session_key(&A, &b, &p);
        let s2 = gen_shared_session_key(&B, &a, &p);

        assert!(s1 == s2, "s1 != s2");
        println!("Challenge 33: Successful!");
    }

    #[derive(Debug, PartialEq)]
    enum WhichSide {
        A,
        B,
        Undef,
    }

    struct Chall2Ctx {
        privkey: BigUint,
        pubkey: BigUint,
        p: BigUint,
        g: BigUint,

        session_key: Option<[u8; 16]>,
        side: WhichSide,
        other_side_pubkey: Option<BigUint>,
    }

    impl Default for Chall2Ctx {
        fn default() -> Self {
            let privkey = gen_privkey();
            let p = BigUint::parse_bytes(P_STR, 16).unwrap();
            let g = BigUint::from(2u32);
            let pubkey = get_pubkey(&p, &g, &privkey);
            let other_side_pubkey = None;

            Self {privkey, pubkey, p, g, session_key: None, side: WhichSide::A, other_side_pubkey}
        }
    }

    impl Chall2Ctx {
        fn new_from_a_stage_1(a_stage_1_res: Vec<&BigUint>) -> Self {
            let p = a_stage_1_res[0].clone();
            let g = a_stage_1_res[1].clone();
            let other_side_pubkey = a_stage_1_res[2].clone();
            let privkey = gen_privkey();
            let pubkey = get_pubkey(&p, &g, &privkey);

            Self {privkey, pubkey, p, g, session_key: None, side: WhichSide::B, other_side_pubkey: Some(other_side_pubkey)}
        }

        /// A->B: Send "p", "g", "A"
        /// .0 = p .1 = g .2 = A
        fn a_stage_1(&self) -> Vec<&BigUint> {
            vec!(&self.p, &self.g, &self.pubkey)
        }

        /// B->A: Send "B"
        fn b_stage_1(&self) -> Vec<&BigUint> {
            vec!(&self.pubkey)
        }

        fn decrypt_msg(key: &[u8], msg: &Vec<u8>) -> Result<Vec<u8>, String> {
            let veclen = msg.len();
            let iv = &msg[veclen-16..];
            helpers::aes_128_cbc_decrypt(&msg[0..veclen-16], &key, iv)
        }

        /// A->B: Send AES-CBC(SHA1(s)[0:16], iv=random(16), msg) + iv
        fn send_msg(&mut self) -> Vec<u8> {
            let sk = gen_shared_session_key(self.other_side_pubkey.as_ref().unwrap(), &self.privkey, &self.p);
            self.session_key = Some(sk);

            let mut iv = [0u8; 16];
            rand::thread_rng().fill_bytes(&mut iv);

            let mut enc = helpers::aes_128_cbc_encrypt(b"This is a test string", &sk, &iv);

            enc.extend_from_slice(&iv);

            enc
        }

        /// B->A: Send AES-CBC(SHA1(s)[0:16], iv=random(16), A's msg) + iv
        fn echo_msg(&mut self, msg: &Vec<u8>) -> Vec<u8> {
            let sk = gen_shared_session_key(self.other_side_pubkey.as_ref().unwrap(), &self.privkey, &self.p);
            self.session_key = Some(sk);

            let dec_data = Chall2Ctx::decrypt_msg(&sk, &msg).unwrap();

            let mut new_iv = [0u8; 16];
            rand::thread_rng().fill_bytes(&mut new_iv);

            let mut enc_data = helpers::aes_128_cbc_encrypt(&dec_data, &sk, &new_iv);

            enc_data.extend_from_slice(&new_iv);
            enc_data
        }

        fn receive_echoed_msg(&self, msg: Vec<u8>) {
            let dec_data = Chall2Ctx::decrypt_msg(self.session_key.as_ref().unwrap(), &msg).unwrap();
            assert!(dec_data == b"This is a test string".to_vec());
        }

        fn stage_1(&self) -> Vec<&BigUint> {
            if self.side == WhichSide::A {
                return self.a_stage_1();
            } else if self.side == WhichSide::B {
                return self.b_stage_1();
            } else {
                panic!("side Undef in stage 1!!\n");
            }
        }

        fn a_receive_msg(&mut self, msg: Vec<&BigUint>) {
            self.other_side_pubkey = Some(msg[0].clone());
            return;
        }

        fn b_receive_msg(&mut self, msg: Vec<&BigUint>) {
            // A->B: Send "p", "g", "A"
            self.p = msg[0].clone();
            self.g = msg[1].clone();
            self.other_side_pubkey = Some(msg[2].clone());
            self.pubkey = get_pubkey(&self.p, &self.g, &self.privkey);
            return;
        }

        fn receive_msg(&mut self, msg: Vec<&BigUint>) {
            match self.side {
                WhichSide::A => self.a_receive_msg(msg),
                WhichSide::B => self.b_receive_msg(msg),
                _ => panic!("side Undef in receive_msg!!\n"),
            }
        }
    }

    pub fn challenge_34() {
        let mut side_a = Chall2Ctx::default(); // side gets set to A by default
        let mut side_b = Chall2Ctx::default();
        side_b.side = WhichSide::B;

        let hardcoded_key: [u8; 16] = helpers::sha1sum(&b"\x00"[..])[0..16].try_into().unwrap();

        let msg = side_a.stage_1();
        let mitm_p = msg[0].clone();
        let msg = vec!(msg[0], msg[1], msg[0]);
        side_b.receive_msg(msg);

        let msg = side_b.stage_1();
        let msg = vec!(&mitm_p);
        side_a.receive_msg(msg);

        // handshake is done now

        let msg = side_a.send_msg();
        let mitm_msg = Chall2Ctx::decrypt_msg(&hardcoded_key, &msg).unwrap();
        assert!(mitm_msg == b"This is a test string".to_vec());
        //println!("decrypted mitm msg: '{}'", String::from_utf8_lossy(&mitm_msg));

        let msg = side_b.echo_msg(&msg);
        let mitm_msg = Chall2Ctx::decrypt_msg(&hardcoded_key, &msg).unwrap();
        assert!(mitm_msg == b"This is a test string".to_vec());
        //println!("decrypted echo mitm msg: '{}'", String::from_utf8_lossy(&mitm_msg));
        side_a.receive_echoed_msg(msg);

        /*
        // Non MITM version
        let msg = side_a.stage_1();
        side_b.receive_msg(msg);
        let msg = side_b.stage_1();
        side_a.receive_msg(msg);

        // handshake is done now

        let msg = side_a.send_msg();
        let msg = side_b.echo_msg(&msg);
        side_a.receive_echoed_msg(msg);
        */

        println!("Challenge 34: Successful!");
    }

    /// not gonna write all new code for the small change of sending an ACK
    /// and THEN "A" instead of sending "A" right away.
    pub fn challenge_35() {
        for i in 0..3 {
            let mut side_a = Chall2Ctx::default(); // side gets set to A by default
            let mut side_b = Chall2Ctx::default();
            side_b.side = WhichSide::B;

            let hardcoded_g = match i {
                0 => BigUint::from(1u32),
                1 => side_a.p.clone(),
                2 => side_a.p.clone() - 1u32,
                _ => panic!("unreachable!"),
            };

            // hacky and ugly cause i dont want to rewrite the logic.
            // still achieves the same effectively.
            side_a.g = hardcoded_g.clone();
            side_a.pubkey = get_pubkey(&side_a.p, &side_a.g, &side_a.privkey);

            let msg = side_a.stage_1();

            //let mitm_p = msg[0].clone();

            let msg = vec!(msg[0], msg[1], msg[2]);
            // "p" "g" "A"
            side_b.receive_msg(msg);

            let hardcoded_shared_secret = match i {
                0 => hardcoded_g.clone(),
                1 => BigUint::from(0u32), //side_a.p.clone(),
                2 => hardcoded_g.clone(),
                _ => panic!("unreachable!"),
            };

            let mut hardcoded_key: [u8; 16] = helpers::sha1sum(&hardcoded_shared_secret.to_bytes_le())[0..16].try_into().unwrap();

            let msg = side_b.stage_1();
            // "B"
            side_a.receive_msg(msg);

            // handshake is done now

            let msg = side_a.send_msg();

            if &hardcoded_g == &(&side_a.p - 1u32) {
                // this case can be either p - 1 or 1 it depends on the secret
                // try one and if that fails try the other
                if let Ok(_) = Chall2Ctx::decrypt_msg(&hardcoded_key, &msg) {
                    // Great we matched, nothing needs to be done
                } else {
                    // the decryption failed, which means the shared secret
                    // was the other option
                    hardcoded_key = helpers::sha1sum(&BigUint::from(1u32).to_bytes_le())[0..16].try_into().unwrap();
                }
            }

            let mitm_msg = Chall2Ctx::decrypt_msg(&hardcoded_key, &msg).unwrap();
            assert!(mitm_msg == b"This is a test string".to_vec());
            //println!("decrypted mitm msg: '{}'", String::from_utf8_lossy(&mitm_msg));

            let msg = side_b.echo_msg(&msg);
            let mitm_msg = Chall2Ctx::decrypt_msg(&hardcoded_key, &msg).unwrap();
            assert!(mitm_msg == b"This is a test string".to_vec());
            //println!("decrypted mitm msg: '{}'", String::from_utf8_lossy(&mitm_msg));
            side_a.receive_echoed_msg(msg);
        }

        
        println!("Challenge 35: Successful!");
    }

    // the Option<> ones are the ones that will get sent by the other side
    #[derive(Debug, PartialEq)]
    struct SrpServer {
        N: BigUint,
        g: BigUint,
        k: BigUint,
        email: String,
        salt: BigUint,
        v: BigUint,
        u: Option<BigUint>,
        privkey: BigUint,
        pubkey: BigUint,
        other_side_pubkey: BigUint,
        S: Option<BigUint>,
        K: Option<[u8; 20]>,
    }

    impl SrpServer {
        fn recv(&mut self, serd: String) {
            let deserd = generic_kv_parse::deserialize(&serd).unwrap();

            for (k, v) in deserd {
                match k {
                    //"email" => self.email = Some(v),
                    "email" => self.email = v,
                    "u" => self.u = Some(BigUint::parse_bytes(v.as_bytes(), 16).unwrap()),
                    //"pubkey" => self.other_side_pubkey = Some(BigUint::parse_bytes(v.as_bytes(), 16).unwrap()),
                    "pubkey" => self.other_side_pubkey = BigUint::parse_bytes(v.as_bytes(), 16).unwrap(),
                    //"v" => self.v = Some(BigUint::parse_bytes(v.as_bytes(), 16).unwrap()),
                    "v" => self.v = BigUint::parse_bytes(v.as_bytes(), 16).unwrap(),
                    _ => panic!("unreachable!"),
                }
            }
        }

        fn register(email: &String, other_side_pubkey: &BigUint, salt: &BigUint, v: &BigUint) -> Self {
            let N = BigUint::parse_bytes(P_STR, 16).unwrap();
            let g = BigUint::from(2u32);
            let k = BigUint::from(3u32);

            let privkey = gen_privkey();
            let pubkey = (&k * v) + &g.modpow(&privkey, &N);

            let mut s = Self {
                N,
                g,
                k,
                email: email.clone(),
                salt: salt.clone(),
                v: v.clone(),
                privkey,
                pubkey,
                other_side_pubkey: other_side_pubkey.clone(),
                u: None,
                S: None,
                K: None,
            };

            s.set_u();
            s
        }

        //fn login(email: &String, other_side_pubkey: &BigUint)

        fn set_u(&mut self) {
            //let mut tmp = self.other_side_pubkey.as_ref().unwrap().to_bytes_le();
            let mut tmp = self.pubkey.to_bytes_le();
            //tmp.extend(self.pubkey.to_bytes_le());
            tmp.extend(self.other_side_pubkey.to_bytes_le());
            let uH = helpers::sha1sum(&tmp);
            self.u = Some(BigUint::from_bytes_le(&uH));
        }

        fn gen_shared_session_key(&mut self) {
            let A = &self.other_side_pubkey;
            let v = &self.v;
            let u = self.u.as_ref().unwrap();
            let privkey = &self.privkey;
            let N = &self.N;
            let v = &self.v;

            let S = (A * v.modpow(u, N)).modpow(privkey, N);

            let K = helpers::sha1sum(&S.to_bytes_le());
            self.K = Some(K);

            self.S = Some(S);
        }

        fn validate_hmac(&self, hmac: &[u8; 20]) -> bool {
            let K = self.K.as_ref().unwrap();
            let salt = self.salt.to_bytes_le();

            helpers::verify_sha1_hmac(hmac, K, &salt)
        }

    }

    // (most of) the Option<> ones are the ones that will get sent by the other side
    #[derive(Debug, PartialEq)]
    struct SrpClient {
        N: BigUint,
        g: BigUint,
        k: BigUint,
        privkey: BigUint,
        pubkey: BigUint,
        email: String,
        password: String,
        salt: BigUint,
        u: Option<BigUint>,
        other_side_pubkey: Option<BigUint>,
        S: Option<BigUint>,
        K: Option<[u8; 20]>,
    }

    impl Default for SrpClient {
        fn default() -> Self {
            let N = BigUint::parse_bytes(P_STR, 16).unwrap();
            let g = BigUint::from(2u32);
            let k = BigUint::from(3u32);
            let email = "test@test.test".to_string();
            let password = "test".to_string();
            let salt = BigUint::from(0xdeadbeefu32);

            let privkey = gen_privkey();
            //let privkey = BigUint::from(7u32);
            let pubkey = get_pubkey(&N, &g, &privkey);

            Self {N, g, k, privkey, pubkey, email, password, salt, u: None,
                other_side_pubkey: None, S: None, K: None}
        }
    }

    impl SrpClient {
        fn recv(&mut self, serd: String) {
            let deserd = generic_kv_parse::deserialize(&serd).unwrap();

            for (k, v) in deserd {
                match k {
                    "pubkey" => self.other_side_pubkey = Some(BigUint::parse_bytes(v.as_bytes(), 16).unwrap()),
                    //"salt" => self.salt = Some(BigUint::parse_bytes(v.as_bytes(), 16).unwrap()),
                    "salt" => self.salt = BigUint::parse_bytes(v.as_bytes(), 16).unwrap(),
                    _ => panic!("unreachable!"),
                }
            }
        }

        fn gen_v(&mut self) -> BigUint {
            let salt = BigUint::from(rand::random::<u32>());
            self.salt = salt;

            let mut tmp = self.salt.to_bytes_le();
            tmp.extend_from_slice(&self.password.as_bytes());
            let xH = helpers::sha1sum(&tmp);
            let x = BigUint::from_bytes_le(&xH);
            self.g.modpow(&x, &self.N)
        }

        fn set_u(&mut self) {
            let mut tmp = self.other_side_pubkey.as_ref().unwrap().to_bytes_le();
            tmp.extend(self.pubkey.to_bytes_le());
            let uH = helpers::sha1sum(&tmp);
            self.u = Some(BigUint::from_bytes_le(&uH));
        }

        fn gen_shared_session_key(&mut self) {
            //use num_traits::Pow;
            let mut tmp = self.salt.to_bytes_le();
            tmp.extend_from_slice(self.password.as_bytes());
            let xH = helpers::sha1sum(&tmp);
            let x = BigUint::from_bytes_le(&xH);

            let B = self.other_side_pubkey.as_ref().unwrap();
            let k = &self.k;
            let g = &self.g;
            let N = &self.N;
            let privkey = &self.privkey;
            let u = self.u.as_ref().unwrap();

            let S = (B - (k * g.modpow(&x, N))).modpow(&(privkey + (u*&x)), N);
            let K = helpers::sha1sum(&S.to_bytes_le());
            self.K = Some(K);

            self.S = Some(S);
        }

        fn gen_hmac(&self) -> [u8; 20] {
            let K = self.K.as_ref().unwrap();
            let salt = self.salt.to_bytes_le();

            helpers::create_sha1_hmac(K, &salt)
        }
    }

    pub fn challenge_36() {
        let mut c = SrpClient {email: "test@test.com".to_string(), password: "password".to_string(), ..Default::default()};
        let v = c.gen_v();

        let mut s = SrpServer::register(&c.email, &c.pubkey, &c.salt, &v);

        // to begin authentication the client sends the email and pubkey
        s.recv(format!("email={}&pubkey={:x}", &c.email, &c.pubkey));

        // the server then responds with the salt and its pubkey
        c.recv(format!("salt={:x}&pubkey={:x}", &s.salt, &s.pubkey));

        // Both server and client create an integer out of the combined
        // public keys
        s.set_u();
        c.set_u();
        assert!(s.u == c.u, "server and client u differs!");

        // Both server and client generate the shared secret and make a session key
        c.gen_shared_session_key();
        s.gen_shared_session_key();
        assert!(c.S == s.S, "Client and Server session keys differ!");

        let hmac = c.gen_hmac();
        assert!(s.validate_hmac(&hmac) == true);

        println!("Challenge 36: SRP Successful!");
        return;
    }

    pub fn challenge_37() {
        let mut A_vals = vec!(BigUint::from(0u32), BigUint::parse_bytes(P_STR, 16).unwrap());
        A_vals.push(&A_vals[1] * 2u32);
        A_vals.push(&A_vals[2] * 2u32);
        A_vals.push(&A_vals[3] * 2u32);

        for i in 0..A_vals.len() {
            let mut c = SrpClient {email: "test@test.com".to_string(), password: "password".to_string(), ..Default::default()};
            let v = c.gen_v();
            let mut s = SrpServer::register(&c.email, &c.pubkey, &c.salt, &v);

            c.password = "wrongpassword".to_string();
            c.pubkey = BigUint::from(0u32);

            // to begin authentication the client sends the email and pubkey
            s.recv(format!("email={}&pubkey={:x}", &c.email, &c.pubkey));

            // the server then responds with the salt and its pubkey
            c.recv(format!("salt={:x}&pubkey={:x}", &s.salt, &s.pubkey));

            // Both server and client create an integer out of the combined
            // public keys
            s.set_u();
            c.set_u();
            assert!(s.u == c.u, "server and client u differs!");

            // Both server and client generate the shared secret and make a session key
            //c.gen_shared_session_key();
            let zero = BigUint::from(0u32);
            let sum = helpers::sha1sum(&zero.to_bytes_le());
            c.K = Some(sum);
            c.S = Some(BigUint::from(0u32));

            s.gen_shared_session_key();
            assert!(c.S == s.S, "Client and Server session keys differ!\nC: {:x}\nS: {:x}", c.S.unwrap(), s.S.unwrap());

            let hmac = c.gen_hmac();
            assert!(s.validate_hmac(&hmac) == true);
        }

        println!("Challenge 37: SRP without password succesful!");
    }

    pub fn challenge_38() {
        return;
    }
}

fn main() {

    println!("Doing set 1");
    set1::challenge_1();
    set1::challenge_2();
    set1::challenge_3();
    set1::challenge_4();
    set1::challenge_5();
    set1::challenge_6();
    set1::challenge_7();
    set1::challenge_8();
    println!("Set 1 complete!\n");

    println!("Doing set 2");
    set2::challenge_9();
    set2::challenge_10();
    set2::challenge_11();
    set2::challenge_12(); // Slow
    set2::challenge_13();
    set2::challenge_14(); // Slow
    set2::challenge_15();
    set2::challenge_16();
    println!("Set 2 complete!\n");

    println!("Doing set 3");
    set3::challenge_17();
    set3::challenge_18();
    set3::challenge_19();
    set3::challenge_20();
    set3::challenge_21();
    set3::challenge_22();
    set3::challenge_23();
    set3::challenge_24();
    println!("Set 3 complete!\n");

    println!("Doing set 4");
    set4::challenge_25();
    set4::challenge_26();
    set4::challenge_27();
    set4::challenge_28();
    set4::challenge_29();
    set4::challenge_30();
    set4::challenge_31();   // Sloow
    println!("Set 4 complete!\n");

    println!("Doing set 5");
    set5::challenge_33();
    set5::challenge_34();
    set5::challenge_35();
    set5::challenge_36();
    set5::challenge_37();
}
