"""
Error-correcting code helpers (BCH and Reed-Solomon) used to encode the
secret message into a binary packet before it is embedded, and to recover
it from noisy decoder output.
"""

import bchlib
from reedsolo import RSCodec




class BCH:
    """
    INFO:
    --------------------------------------------------
    INPUT:
    --------------------------------------------------
    OUTPUT:
    """
    def __init__(BCH_BITS, BCH_POLYNOMIAL=137, number_zeros=4, bits_size=100):
        self.BCH_BITS = BCH_BITS
        self.BCH_POLYNOMIAL = BCH_POLYNOMIAL
        self.number_zeros = number_zeros
        self.bits_size = bits_size
        self.bch = bchlib.BCH(self.BCH_POLYNOMIAL, self.BCH_BITS)
    def Generator(self,secret, sevensize = 1 ):  
        #print(tcolors.RED, secret, tcolors.ENDC)

        data = bytearray(secret + ' '*(sevensize-len(secret)), 'utf-8')
        ecc = self.bch.encode(data)
        packet = data + ecc
        packet_binary = ''.join(format(x, '08b') for x in packet)
        secret = [int(x) for x in packet_binary]
        secret.extend([0 for i in range(number_zeros)])
        return secret
    def Reader(self, secret):
        bits = self.bits_size - self.number_zeros
        
        packet_binary = "".join([str(int(bit)) for bit in secret[:bits]])
        packet = bytes(int(packet_binary[i : i + 8], 2) for i in range(0, len(packet_binary), 8))
        packet = bytearray(packet)
    
    
        data, ecc = packet[:-bch.ecc_bytes], packet[-bch.ecc_bytes:]
        bitflips = self.bch.decode_inplace(data, ecc)
  
        try:
            code = data.decode("utf-8")
            print("\n message:",code)
            return code
        except:
            print( "Fail to decoder")
            return "none"
            
class ReadSolomon:
    """
    INFO:
    --------------------------------------------------
    INPUT:
    --------------------------------------------------
    OUTPUT:
    """
    
    def __init__(self, ecc_symbols ,number_zeros = 4, bits_size =100):
        self.ecc_symbols = ecc_symbols
        self.bits_size = bits_size
        self.number_zeros = number_zeros
    def Generator(self, secret ,sevensize =1):
        
        rsc = RSCodec(self.ecc_symbols)
        data = bytearray(secret + ' '*(sevensize-len(secret)), 'utf-8')
        encoded_messsage = rsc.encode(data)
        packet = encoded_messsage

        print(packet)
        packet_binary = ''.join(format(x, '08b') for x in packet)
        secret = [int(x) for x in packet_binary]
        secret.extend([0 for i in range(self.number_zeros)])
        return secret
        
    def Reader(self, secret):
        bits_size = self.bits_size - self.number_zeros
        packet_binary = "".join([str(int(bit)) for bit in secret[:bits_size]])
        packet = bytes(int(packet_binary[i : i + 8], 2) for i in range(0, len(packet_binary), 8))
        #packet = bytearray(packet)
        rsc = RSCodec(self.ecc_symbols)
        try:
            decoded_msg, decoded_msgecc, errata_pos = rsc.decode(packet)
            decoded_msg = decoded_msg.decode()
            print(tcolors.GREEN, decoded_msg, tcolors.ENDC)
        except:
            decoded_msg = "Error reading Code" +  "       " + "BinaryCode: " + packet_binary
            print(tcolors.RED, decoded_msg, tcolors.ENDC)
        return decoded_msg
